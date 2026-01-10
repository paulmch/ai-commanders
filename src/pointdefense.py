"""
Point Defense Laser System for the AI Commanders space battle simulator.

This module implements point defense laser mechanics including:
- Diffraction-limited spot size calculations
- Intensity and ablation rate computations
- Slug evaporation mechanics
- Torpedo heat damage models
- Close-range ship targeting capabilities

Based on realistic laser physics with parameters from Terra Invicta game data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Physical constants
PI = math.pi


class TargetMaterial(Enum):
    """Material types with their vaporization energies."""
    STEEL = "steel"
    TUNGSTEN = "tungsten"
    ALUMINUM = "aluminum"
    TITANIUM = "titanium"


# Material vaporization energies in MJ/kg (approximate values)
# Includes energy to heat, melt, and vaporize
MATERIAL_VAPORIZATION_ENERGY: dict[TargetMaterial, float] = {
    TargetMaterial.STEEL: 30.0,      # ~30 MJ/kg for steel
    TargetMaterial.TUNGSTEN: 60.0,   # ~60 MJ/kg for tungsten (high melting point)
    TargetMaterial.ALUMINUM: 15.0,   # ~15 MJ/kg for aluminum
    TargetMaterial.TITANIUM: 25.0,   # ~25 MJ/kg for titanium
}

# Torpedo damage thresholds in joules
TORPEDO_ELECTRONICS_THRESHOLD_J = 10_000.0   # 10 kJ - electronics fail
TORPEDO_WARHEAD_THRESHOLD_J = 100_000.0      # 100 kJ - warhead detonates


@dataclass
class PDLaser:
    """
    Point Defense Laser system with diffraction-limited optics.

    Attributes:
        power_mw: Laser power output in megawatts.
        aperture_m: Primary mirror/lens diameter in meters.
        wavelength_nm: Laser wavelength in nanometers.
        range_km: Maximum effective range in kilometers.
        cooldown_s: Time between shots/bursts in seconds.
        name: Display name of the PD laser system.
    """
    power_mw: float = 5.0
    aperture_m: float = 0.5
    wavelength_nm: float = 1000.0  # 1 micron IR laser
    range_km: float = 100.0
    cooldown_s: float = 0.5
    name: str = "PD Laser Turret"

    @classmethod
    def from_fleet_data(cls, weapon_data: dict) -> PDLaser:
        """
        Create a PDLaser instance from fleet data JSON.

        Args:
            weapon_data: Dictionary containing PD laser specifications.

        Returns:
            A configured PDLaser instance.
        """
        return cls(
            power_mw=weapon_data.get("power_draw_mw", 5.0),
            aperture_m=weapon_data.get("aperture_m", 0.5),
            wavelength_nm=weapon_data.get("wavelength_nm", 1000.0),
            range_km=weapon_data.get("range_km", 100.0),
            cooldown_s=weapon_data.get("cooldown_s", 0.5),
            name=weapon_data.get("name", "PD Laser Turret"),
        )

    @property
    def wavelength_m(self) -> float:
        """Get wavelength in meters."""
        return self.wavelength_nm * 1e-9

    @property
    def power_w(self) -> float:
        """Get power in watts."""
        return self.power_mw * 1e6

    def calculate_spot_size(self, distance_km: float) -> float:
        """
        Calculate the diffraction-limited spot size at a given distance.

        Uses the Rayleigh criterion: spot_diameter = 2.44 * wavelength * distance / aperture
        Simplified here to: spot_diameter ~ wavelength * distance / aperture

        For practical point defense calculations, we use the first-order approximation
        where most energy is concentrated.

        Args:
            distance_km: Distance to target in kilometers.

        Returns:
            Spot diameter in meters.
        """
        distance_m = distance_km * 1000.0
        # Diffraction limit formula: theta = lambda / D
        # spot_size = theta * distance = (lambda / D) * distance
        spot_diameter = (self.wavelength_m / self.aperture_m) * distance_m
        return spot_diameter

    def calculate_spot_area(self, distance_km: float) -> float:
        """
        Calculate the spot area at a given distance.

        Args:
            distance_km: Distance to target in kilometers.

        Returns:
            Spot area in square meters.
        """
        diameter = self.calculate_spot_size(distance_km)
        return PI * (diameter / 2.0) ** 2

    def calculate_intensity(self, distance_km: float) -> float:
        """
        Calculate beam intensity (power density) at a given distance.

        Intensity follows inverse square law through the spot area.

        Args:
            distance_km: Distance to target in kilometers.

        Returns:
            Intensity in W/m^2.
        """
        spot_area = self.calculate_spot_area(distance_km)
        if spot_area <= 0:
            return 0.0
        return self.power_w / spot_area

    def effectiveness_factor(self, distance_km: float) -> float:
        """
        Calculate range effectiveness factor.

        Effectiveness follows 1/r^2 relationship due to beam spreading.
        Normalized so that effectiveness = 1.0 at 1 km.

        Args:
            distance_km: Distance to target in kilometers.

        Returns:
            Effectiveness factor (higher is better).
        """
        if distance_km <= 0:
            return float('inf')
        # Normalize to 1.0 at 1 km reference distance
        return 1.0 / (distance_km ** 2)

    def is_in_range(self, distance_km: float) -> bool:
        """
        Check if target is within effective range.

        Args:
            distance_km: Distance to target in kilometers.

        Returns:
            True if target is in range.
        """
        return 0 < distance_km <= self.range_km

    def calculate_ablation_rate(
        self,
        distance_km: float,
        material: TargetMaterial = TargetMaterial.STEEL
    ) -> float:
        """
        Calculate the mass ablation rate for a target material.

        ablation_rate (kg/s) = power (W) / vaporization_energy (J/kg)

        This is adjusted by the spot area to account for energy spreading.

        Args:
            distance_km: Distance to target in kilometers.
            material: Target material type.

        Returns:
            Ablation rate in kg/s.
        """
        vaporization_energy_j_per_kg = MATERIAL_VAPORIZATION_ENERGY[material] * 1e6
        # Power delivered to spot
        intensity = self.calculate_intensity(distance_km)
        spot_area = self.calculate_spot_area(distance_km)

        # For ablation, we consider power absorbed by the target
        # Assume target absorbs all incident energy on the spot
        power_delivered = self.power_w

        # Ablation rate = power / energy per kg
        return power_delivered / vaporization_energy_j_per_kg

    def time_to_ablate_mass(
        self,
        mass_kg: float,
        distance_km: float,
        material: TargetMaterial = TargetMaterial.STEEL
    ) -> float:
        """
        Calculate time required to completely ablate a given mass.

        Args:
            mass_kg: Target mass in kilograms.
            distance_km: Distance to target in kilometers.
            material: Target material type.

        Returns:
            Time in seconds to ablate the mass.
        """
        ablation_rate = self.calculate_ablation_rate(distance_km, material)
        if ablation_rate <= 0:
            return float('inf')
        return mass_kg / ablation_rate

    def shots_to_destroy_slug(
        self,
        slug_mass_kg: float,
        distance_km: float,
        material: TargetMaterial = TargetMaterial.STEEL
    ) -> int:
        """
        Calculate number of shots needed to completely vaporize a kinetic slug.

        Each "shot" is one cooldown period of continuous firing.

        Args:
            slug_mass_kg: Mass of the kinetic projectile in kg.
            distance_km: Distance to target in kilometers.
            material: Slug material type.

        Returns:
            Number of shots (cooldown periods) needed.
        """
        time_to_destroy = self.time_to_ablate_mass(slug_mass_kg, distance_km, material)
        # Each shot is one cooldown period of firing
        shots = math.ceil(time_to_destroy / self.cooldown_s)
        return max(1, shots)


@dataclass
class Torpedo:
    """
    Torpedo target for point defense engagement.

    Attributes:
        mass_kg: Total torpedo mass in kilograms.
        thermal_threshold_j: Energy to disable electronics in joules.
        warhead_threshold_j: Energy to detonate warhead in joules.
        heat_absorbed_j: Accumulated heat damage in joules.
        is_active: Whether torpedo is still functional.
    """
    mass_kg: float = 1600.0
    thermal_threshold_j: float = TORPEDO_ELECTRONICS_THRESHOLD_J
    warhead_threshold_j: float = TORPEDO_WARHEAD_THRESHOLD_J
    heat_absorbed_j: float = 0.0
    is_active: bool = True

    def absorb_heat(self, energy_j: float) -> None:
        """
        Absorb heat energy from laser damage.

        Args:
            energy_j: Energy absorbed in joules.
        """
        self.heat_absorbed_j += energy_j
        # Check if disabled or destroyed
        if self.heat_absorbed_j >= self.thermal_threshold_j:
            self.is_active = False

    def is_disabled(self) -> bool:
        """
        Check if torpedo electronics are disabled.

        Returns:
            True if heat absorbed exceeds thermal threshold.
        """
        return self.heat_absorbed_j >= self.thermal_threshold_j

    def is_destroyed(self) -> bool:
        """
        Check if torpedo warhead has detonated (destroyed).

        Returns:
            True if heat absorbed exceeds warhead threshold.
        """
        return self.heat_absorbed_j >= self.warhead_threshold_j


@dataclass
class Slug:
    """
    Kinetic projectile (slug) target for point defense.

    Attributes:
        mass_kg: Slug mass in kilograms.
        material: Slug material type.
        mass_ablated_kg: Amount of mass already ablated.
    """
    mass_kg: float
    material: TargetMaterial = TargetMaterial.STEEL
    mass_ablated_kg: float = 0.0

    @property
    def remaining_mass_kg(self) -> float:
        """Get remaining mass after ablation."""
        return max(0.0, self.mass_kg - self.mass_ablated_kg)

    def is_destroyed(self) -> bool:
        """Check if slug has been completely vaporized."""
        return self.remaining_mass_kg <= 0.0

    def ablate(self, mass_kg: float) -> float:
        """
        Ablate mass from the slug.

        Args:
            mass_kg: Mass to ablate in kilograms.

        Returns:
            Actual mass ablated (may be less if slug is destroyed).
        """
        actual_ablation = min(mass_kg, self.remaining_mass_kg)
        self.mass_ablated_kg += actual_ablation
        return actual_ablation


@dataclass
class ShipArmorTarget:
    """
    Ship armor target for close-range PD laser engagement.

    Attributes:
        armor_thickness_cm: Armor thickness in centimeters.
        armor_type: Type of armor material.
        surface_temperature_k: Current surface temperature in Kelvin.
    """
    armor_thickness_cm: float
    armor_type: TargetMaterial = TargetMaterial.TITANIUM
    surface_temperature_k: float = 300.0  # Ambient starting temperature

    # Armor thermal properties (approximate)
    # Specific heat capacity in J/(kg*K)
    SPECIFIC_HEAT = 500.0
    # Surface density in kg/m^2 per cm thickness
    SURFACE_DENSITY_PER_CM = 48.2  # Titanium: 4820 kg/m^3 / 100

    @property
    def surface_mass_per_m2(self) -> float:
        """Get surface mass per square meter."""
        return self.armor_thickness_cm * self.SURFACE_DENSITY_PER_CM


class EngagementOutcome(Enum):
    """Possible outcomes of a point defense engagement."""
    DESTROYED = "destroyed"
    DISABLED = "disabled"
    DAMAGED = "damaged"
    MISSED = "missed"
    OUT_OF_RANGE = "out_of_range"
    INEFFECTIVE = "ineffective"


@dataclass
class EngagementResult:
    """
    Result of a point defense engagement.

    Attributes:
        outcome: The engagement outcome.
        target_type: Type of target engaged ("slug", "torpedo", "ship").
        distance_km: Engagement distance in kilometers.
        energy_delivered_j: Total energy delivered to target.
        dwell_time_s: Time on target in seconds.
        shots_fired: Number of shots/bursts fired.
        mass_ablated_kg: For slugs, mass vaporized.
        heat_absorbed_j: For torpedoes, heat damage dealt.
        details: Additional engagement details.
    """
    outcome: EngagementOutcome
    target_type: str
    distance_km: float
    energy_delivered_j: float = 0.0
    dwell_time_s: float = 0.0
    shots_fired: int = 0
    mass_ablated_kg: float = 0.0
    heat_absorbed_j: float = 0.0
    details: str = ""

    def __str__(self) -> str:
        """Human-readable engagement result."""
        if self.outcome == EngagementOutcome.OUT_OF_RANGE:
            return f"PD engagement failed: {self.target_type} at {self.distance_km:.1f} km out of range"

        outcome_str = self.outcome.value.upper()
        if self.target_type == "slug":
            return (
                f"PD vs slug at {self.distance_km:.1f} km: {outcome_str}, "
                f"{self.shots_fired} shots, {self.mass_ablated_kg:.2f} kg ablated"
            )
        elif self.target_type == "torpedo":
            return (
                f"PD vs torpedo at {self.distance_km:.1f} km: {outcome_str}, "
                f"{self.dwell_time_s:.2f}s dwell, {self.heat_absorbed_j/1000:.1f} kJ absorbed"
            )
        else:
            return (
                f"PD vs {self.target_type} at {self.distance_km:.1f} km: {outcome_str}, "
                f"{self.energy_delivered_j/1e6:.2f} MJ delivered"
            )


@dataclass
class PDEngagement:
    """
    Point Defense engagement controller.

    Manages engagements between a PD laser and various target types.

    Attributes:
        laser: The point defense laser system.
    """
    laser: PDLaser

    def engage_slug(
        self,
        slug: Slug,
        distance_km: float,
        max_shots: Optional[int] = None
    ) -> EngagementResult:
        """
        Engage a kinetic slug with the PD laser.

        Attempts to vaporize the slug through sustained laser fire.

        Args:
            slug: The kinetic projectile to engage.
            distance_km: Distance to target in kilometers.
            max_shots: Maximum number of shots to fire (None for unlimited).

        Returns:
            EngagementResult with outcome details.
        """
        if not self.laser.is_in_range(distance_km):
            return EngagementResult(
                outcome=EngagementOutcome.OUT_OF_RANGE,
                target_type="slug",
                distance_km=distance_km,
            )

        # Calculate ablation per shot (one cooldown period of firing)
        ablation_rate = self.laser.calculate_ablation_rate(distance_km, slug.material)
        mass_per_shot = ablation_rate * self.laser.cooldown_s

        shots_needed = self.laser.shots_to_destroy_slug(
            slug.remaining_mass_kg, distance_km, slug.material
        )

        if max_shots is not None:
            shots_to_fire = min(shots_needed, max_shots)
        else:
            shots_to_fire = shots_needed

        # Apply damage
        total_ablated = 0.0
        for _ in range(shots_to_fire):
            ablated = slug.ablate(mass_per_shot)
            total_ablated += ablated
            if slug.is_destroyed():
                break

        energy_delivered = total_ablated * MATERIAL_VAPORIZATION_ENERGY[slug.material] * 1e6
        dwell_time = shots_to_fire * self.laser.cooldown_s

        if slug.is_destroyed():
            outcome = EngagementOutcome.DESTROYED
        elif total_ablated > 0:
            outcome = EngagementOutcome.DAMAGED
        else:
            outcome = EngagementOutcome.INEFFECTIVE

        return EngagementResult(
            outcome=outcome,
            target_type="slug",
            distance_km=distance_km,
            energy_delivered_j=energy_delivered,
            dwell_time_s=dwell_time,
            shots_fired=shots_to_fire,
            mass_ablated_kg=total_ablated,
        )

    def engage_torpedo(
        self,
        torpedo: Torpedo,
        distance_km: float,
        dwell_time_s: float
    ) -> EngagementResult:
        """
        Engage a torpedo with the PD laser.

        Attempts to disable or destroy torpedo through heat damage.

        Args:
            torpedo: The torpedo to engage.
            distance_km: Distance to target in kilometers.
            dwell_time_s: Time to keep laser on target in seconds.

        Returns:
            EngagementResult with outcome details.
        """
        if not self.laser.is_in_range(distance_km):
            return EngagementResult(
                outcome=EngagementOutcome.OUT_OF_RANGE,
                target_type="torpedo",
                distance_km=distance_km,
            )

        # Calculate heat transfer
        heat_delivered = self.calculate_heat_transfer(
            self.laser.power_w, distance_km, dwell_time_s
        )

        initial_heat = torpedo.heat_absorbed_j
        torpedo.absorb_heat(heat_delivered)

        shots = math.ceil(dwell_time_s / self.laser.cooldown_s)

        if torpedo.is_destroyed():
            outcome = EngagementOutcome.DESTROYED
        elif torpedo.is_disabled():
            outcome = EngagementOutcome.DISABLED
        elif heat_delivered > 0:
            outcome = EngagementOutcome.DAMAGED
        else:
            outcome = EngagementOutcome.INEFFECTIVE

        return EngagementResult(
            outcome=outcome,
            target_type="torpedo",
            distance_km=distance_km,
            energy_delivered_j=heat_delivered,
            dwell_time_s=dwell_time_s,
            shots_fired=shots,
            heat_absorbed_j=torpedo.heat_absorbed_j - initial_heat,
        )

    def engage_ship(
        self,
        ship_armor: ShipArmorTarget,
        distance_km: float,
        dwell_time_s: float
    ) -> EngagementResult:
        """
        Engage ship armor with the PD laser.

        At close range, PD lasers can heat ship armor but are far less
        effective than kinetic weapons.

        Args:
            ship_armor: The ship armor section to engage.
            distance_km: Distance to target in kilometers.
            dwell_time_s: Time to keep laser on target in seconds.

        Returns:
            EngagementResult with outcome details.
        """
        if not self.laser.is_in_range(distance_km):
            return EngagementResult(
                outcome=EngagementOutcome.OUT_OF_RANGE,
                target_type="ship",
                distance_km=distance_km,
            )

        # Calculate energy delivered
        energy_delivered = self.laser.power_w * dwell_time_s

        # Calculate heating rate and final temperature
        heating_rate = self.calculate_armor_heating_rate(ship_armor, distance_km)
        temperature_rise = heating_rate * dwell_time_s
        ship_armor.surface_temperature_k += temperature_rise

        # Check if we can damage armor
        can_damage = self.can_damage_ship_armor(distance_km, ship_armor.armor_thickness_cm)

        shots = math.ceil(dwell_time_s / self.laser.cooldown_s)

        if can_damage:
            outcome = EngagementOutcome.DAMAGED
            details = f"Armor heated to {ship_armor.surface_temperature_k:.0f} K"
        else:
            outcome = EngagementOutcome.INEFFECTIVE
            details = "PD laser insufficient to damage heavy armor at this range"

        return EngagementResult(
            outcome=outcome,
            target_type="ship",
            distance_km=distance_km,
            energy_delivered_j=energy_delivered,
            dwell_time_s=dwell_time_s,
            shots_fired=shots,
            details=details,
        )

    def calculate_heat_transfer(
        self,
        power_w: float,
        distance_km: float,
        exposure_time_s: float
    ) -> float:
        """
        Calculate heat energy transferred to target.

        Args:
            power_w: Laser power in watts.
            distance_km: Distance to target in kilometers.
            exposure_time_s: Exposure time in seconds.

        Returns:
            Heat energy in joules.
        """
        # At the target, all power is absorbed for the exposure duration
        # Energy = Power * Time
        # But intensity drops with distance (spot spreads)
        # We use a coupling efficiency based on spot vs target size

        # For missiles/torpedoes, assume target absorbs all energy in spot
        # since they're smaller than typical spot sizes
        intensity = self.laser.calculate_intensity(distance_km)
        spot_area = self.laser.calculate_spot_area(distance_km)

        # Assume torpedo cross-section ~1 m^2
        torpedo_cross_section = 1.0  # m^2

        if spot_area <= torpedo_cross_section:
            # Spot smaller than target - all energy hits
            effective_power = power_w
        else:
            # Spot larger than target - only fraction hits
            effective_power = power_w * (torpedo_cross_section / spot_area)

        return effective_power * exposure_time_s

    def can_damage_ship_armor(
        self,
        distance_km: float,
        armor_thickness_cm: float
    ) -> bool:
        """
        Check if PD laser can effectively damage ship armor at given range.

        PD lasers are designed for missiles, not ships. At very close range
        they can heat armor but effectiveness drops rapidly with armor thickness.

        Args:
            distance_km: Distance to target in kilometers.
            armor_thickness_cm: Armor thickness in centimeters.

        Returns:
            True if laser can meaningfully damage the armor.
        """
        # Calculate intensity at range
        intensity = self.laser.calculate_intensity(distance_km)

        # Very rough threshold: need >1 MW/m^2 to start ablating armor
        # and effectiveness drops with armor thickness
        min_intensity = 1e6  # 1 MW/m^2

        # Armor acts as heat sink - thicker armor harder to damage
        thickness_factor = 1.0 / (1.0 + armor_thickness_cm / 10.0)

        effective_intensity = intensity * thickness_factor

        return effective_intensity >= min_intensity

    def calculate_armor_heating_rate(
        self,
        ship_armor: ShipArmorTarget,
        distance_km: float
    ) -> float:
        """
        Calculate armor surface heating rate in degrees per second.

        Args:
            ship_armor: The armor section being heated.
            distance_km: Distance to target in kilometers.

        Returns:
            Heating rate in K/s (Kelvin per second).
        """
        intensity = self.laser.calculate_intensity(distance_km)
        spot_area = self.laser.calculate_spot_area(distance_km)

        # Mass of armor being heated (spot area * thickness-dependent mass)
        # Assume heating penetrates ~1cm for PD lasers
        heated_depth_cm = min(1.0, ship_armor.armor_thickness_cm)
        heated_mass_per_m2 = heated_depth_cm * ship_armor.SURFACE_DENSITY_PER_CM
        heated_mass = heated_mass_per_m2 * spot_area

        if heated_mass <= 0:
            return 0.0

        # Power absorbed
        power_absorbed = self.laser.power_w

        # dT/dt = P / (m * c)
        heating_rate = power_absorbed / (heated_mass * ship_armor.SPECIFIC_HEAT)

        return heating_rate


def is_torpedo_disabled(heat_absorbed_j: float) -> bool:
    """
    Check if torpedo electronics are disabled by heat.

    Args:
        heat_absorbed_j: Heat energy absorbed in joules.

    Returns:
        True if heat exceeds electronics threshold.
    """
    return heat_absorbed_j >= TORPEDO_ELECTRONICS_THRESHOLD_J


def is_torpedo_destroyed(heat_absorbed_j: float) -> bool:
    """
    Check if torpedo is destroyed (warhead detonated) by heat.

    Args:
        heat_absorbed_j: Heat energy absorbed in joules.

    Returns:
        True if heat exceeds warhead threshold.
    """
    return heat_absorbed_j >= TORPEDO_WARHEAD_THRESHOLD_J


def calculate_heat_transfer(
    power_w: float,
    distance_km: float,
    exposure_time_s: float,
    target_cross_section_m2: float = 1.0,
    laser_aperture_m: float = 0.5,
    wavelength_nm: float = 1000.0
) -> float:
    """
    Standalone function to calculate heat transferred to a target.

    Args:
        power_w: Laser power in watts.
        distance_km: Distance to target in kilometers.
        exposure_time_s: Exposure time in seconds.
        target_cross_section_m2: Target cross-sectional area in m^2.
        laser_aperture_m: Laser aperture diameter in meters.
        wavelength_nm: Laser wavelength in nanometers.

    Returns:
        Heat energy transferred in joules.
    """
    # Create temporary laser for calculation
    laser = PDLaser(
        power_mw=power_w / 1e6,
        aperture_m=laser_aperture_m,
        wavelength_nm=wavelength_nm
    )

    spot_area = laser.calculate_spot_area(distance_km)

    if spot_area <= target_cross_section_m2:
        effective_power = power_w
    else:
        effective_power = power_w * (target_cross_section_m2 / spot_area)

    return effective_power * exposure_time_s


if __name__ == "__main__":
    # Example usage and basic validation
    print("AI Commanders Point Defense System")
    print("=" * 50)

    # Create PD laser from typical stats
    pd_laser = PDLaser(
        power_mw=5.0,
        aperture_m=0.5,
        wavelength_nm=1000.0,
        range_km=100.0,
        cooldown_s=0.5
    )

    print(f"\nPD Laser: {pd_laser.name}")
    print(f"  Power: {pd_laser.power_mw} MW")
    print(f"  Aperture: {pd_laser.aperture_m} m")
    print(f"  Wavelength: {pd_laser.wavelength_nm} nm")
    print(f"  Range: {pd_laser.range_km} km")
    print(f"  Cooldown: {pd_laser.cooldown_s} s")

    # Test spot size at various distances
    print("\nSpot Size vs Distance:")
    for dist in [1, 10, 50, 100]:
        spot = pd_laser.calculate_spot_size(dist)
        intensity = pd_laser.calculate_intensity(dist)
        print(f"  {dist:3d} km: {spot*100:.2f} cm diameter, {intensity/1e6:.2f} MW/m^2")

    # Test slug engagement
    print("\nSlug Engagement Test:")
    engagement = PDEngagement(pd_laser)
    test_slug = Slug(mass_kg=50.0, material=TargetMaterial.STEEL)

    for dist in [10, 50, 100]:
        slug_copy = Slug(mass_kg=50.0, material=TargetMaterial.STEEL)
        result = engagement.engage_slug(slug_copy, dist)
        print(f"  At {dist} km: {result.shots_fired} shots to destroy 50 kg steel slug")

    # Test torpedo engagement
    print("\nTorpedo Engagement Test:")
    for dist in [10, 50, 100]:
        torpedo = Torpedo()
        result = engagement.engage_torpedo(torpedo, dist, dwell_time_s=2.0)
        print(f"  At {dist} km, 2s dwell: {result.heat_absorbed_j/1000:.1f} kJ, {result.outcome.value}")
