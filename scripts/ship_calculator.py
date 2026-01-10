#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Ship Mass/Delta-V/Acceleration Calculator for AI Commanders

Based on Terra Invicta reference data. Calculates how armor mass
affects the 4g combat acceleration requirement.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

# =============================================================================
# CONSTANTS
# =============================================================================

G = 9.81  # m/s^2
TARGET_ACCEL_G = 4.0  # Target combat acceleration in g


# =============================================================================
# MODULE DEFINITIONS (from Terra Invicta reference)
# =============================================================================

@dataclass
class Drive:
    name: str
    thrust_mn: float  # Meganewtons
    exhaust_vel_kms: float  # km/s
    mass_tons: float
    efficiency: float = 0.925
    self_powered: bool = False  # If True, generates own power during thrust

    @property
    def thrust_n(self) -> float:
        return self.thrust_mn * 1e6

    @property
    def exhaust_vel_ms(self) -> float:
        return self.exhaust_vel_kms * 1000


# =============================================================================
# TERRA INVICTA DRIVE DATABASE (Human Tech, No Alien/Antimatter)
# =============================================================================
# Drives with flatMass=0 are massless modules - reactor provides the mass
# Drives with specificPower>0 have calculated mass from thrust rating
# required_reactor: which reactor class is needed

@dataclass
class DriveSpec:
    name: str
    thrust_mn: float
    exhaust_vel_kms: float
    drive_mass_tons: float  # 0 if massless (mass from reactor)
    efficiency: float
    required_reactor: str  # Reactor class required

    @property
    def thrust_n(self) -> float:
        return self.thrust_mn * 1e6

    @property
    def exhaust_vel_ms(self) -> float:
        return self.exhaust_vel_kms * 1000


DRIVES = {
    # === INERTIAL CONFINEMENT FUSION (requires ICF reactor) ===
    # Best combat drives - high thrust + high Isp
    'protium_nova_x6': DriveSpec("Protium Nova Torch x6", 39.6, 1000, 0, 0.97, 'icf'),
    'protium_nova_x5': DriveSpec("Protium Nova Torch x5", 33.0, 1000, 0, 0.97, 'icf'),
    'protium_nova_x4': DriveSpec("Protium Nova Torch x4", 26.4, 1000, 0, 0.97, 'icf'),
    # Insane Isp versions
    'protium_converter_x6': DriveSpec("Protium Converter Torch x6", 58.6, 10256, 0, 0.98, 'icf'),
    'protium_converter_x4': DriveSpec("Protium Converter Torch x4", 39.0, 10256, 0, 0.98, 'icf'),

    # === HYBRID CONFINEMENT FUSION (requires Hybrid reactor) ===
    'borane_plasmajet_x6': DriveSpec("Borane Plasmajet Torch x6", 30.2, 714, 0, 0.95, 'hybrid'),
    'borane_plasmajet_x5': DriveSpec("Borane Plasmajet Torch x5", 25.2, 714, 0, 0.95, 'hybrid'),
    'borane_plasmajet_x4': DriveSpec("Borane Plasmajet Torch x4", 20.2, 714, 0, 0.95, 'hybrid'),

    # === TOROID MAGNETIC (Tokamak) ===
    'protium_torus_x6': DriveSpec("Protium Torus Lantern x6", 10.1, 952, 0, 0.95, 'tokamak'),
    'helion_torus_x6': DriveSpec("Helion Torus Lantern x6", 3.34, 690, 0, 0.925, 'tokamak'),

    # === ANY REACTOR (Nuclear Salt Water - self-contained but MASSIVE) ===
    'neutron_flux_x6': DriveSpec("Neutron Flux Torch x6", 78.0, 1700, 9614, 0.80, 'any'),
    'neutron_flux_x5': DriveSpec("Neutron Flux Torch x5", 65.0, 1700, 8011, 0.80, 'any'),
    'neutron_flux_x4': DriveSpec("Neutron Flux Torch x4", 52.0, 1700, 6409, 0.80, 'any'),
}


# =============================================================================
# REACTOR DATABASE WITH CLASS TAGS
# =============================================================================

@dataclass
class ReactorSpec:
    name: str
    output_gw: float
    mass_tons: float
    efficiency: float
    reactor_class: str  # icf, hybrid, tokamak, zpinch, any


REACTORS = {
    # === INERTIAL CONFINEMENT FUSION ===
    'icf_iv': ReactorSpec("Inertial Confinement Fusion IV", 5500, 2750, 0.95, 'icf'),
    'icf_iii': ReactorSpec("Inertial Confinement Fusion III", 3170, 3170, 0.92, 'icf'),
    'icf_ii': ReactorSpec("Inertial Confinement Fusion II", 860, 1720, 0.89, 'icf'),

    # === HYBRID CONFINEMENT FUSION ===
    'hybrid_iv': ReactorSpec("Hybrid Confinement Fusion IV", 11370, 568, 0.99, 'hybrid'),
    'hybrid_iii': ReactorSpec("Hybrid Confinement Fusion III", 1900, 950, 0.99, 'hybrid'),
    'hybrid_ii': ReactorSpec("Hybrid Confinement Fusion II", 510, 510, 0.98, 'hybrid'),

    # === TOKAMAK (Toroid Magnetic) ===
    'tokamak_iv': ReactorSpec("Fusion Tokamak IV", 1260, 630, 0.985, 'tokamak'),
    'tokamak_iii': ReactorSpec("Fusion Tokamak III", 624, 624, 0.96, 'tokamak'),

    # === Z-PINCH FUSION ===
    'zpinch_iv': ReactorSpec("Z-Pinch Fusion IV", 3970, 1588, 0.98, 'zpinch'),
    'zpinch_iii': ReactorSpec("Z-Pinch Fusion III", 2510, 3514, 0.96, 'zpinch'),
}


def get_compatible_reactors(drive: DriveSpec) -> list:
    """Get list of reactors compatible with a drive."""
    if drive.required_reactor == 'any':
        return list(REACTORS.values())
    return [r for r in REACTORS.values() if r.reactor_class == drive.required_reactor]


def get_best_reactor_for_drive(drive: DriveSpec) -> ReactorSpec:
    """Get the lightest compatible reactor for a drive."""
    compatible = get_compatible_reactors(drive)
    if not compatible:
        raise ValueError(f"No compatible reactor for {drive.name}")
    # Return lightest reactor
    return min(compatible, key=lambda r: r.mass_tons)


@dataclass
class Radiator:
    name: str
    specific_power_kw_kg: float
    mass_tons: float

    @property
    def dissipation_mw(self) -> float:
        return self.specific_power_kw_kg * self.mass_tons * 1000 / 1000


@dataclass
class Heatsink:
    name: str
    capacity_gj: float
    mass_tons: float


@dataclass
class Battery:
    name: str
    capacity_gj: float
    mass_tons: float


@dataclass
class Weapon:
    name: str
    mass_tons: float
    ammo_mass_kg: float = 0
    magazine_size: int = 0

    @property
    def total_ammo_tons(self) -> float:
        return (self.ammo_mass_kg * self.magazine_size) / 1000


@dataclass
class ArmorZone:
    zone: str  # 'nose', 'lateral', 'tail'
    material: str
    points: int  # 1 point = 1 cm thickness
    area_m2: float
    density_kg_m3: float

    @property
    def thickness_m(self) -> float:
        return self.points * 0.01

    @property
    def mass_tons(self) -> float:
        volume_m3 = self.area_m2 * self.thickness_m
        mass_kg = volume_m3 * self.density_kg_m3
        return mass_kg / 1000


# Armor material densities (kg/m^3)
ARMOR_MATERIALS = {
    'steel': 7850,
    'titanium': 4820,
    'silicon_carbide': 3210,
    'boron_carbide': 2520,
    'composite': 1930,
    'foamed_metal': 920,
    'nanotube': 1720,
    'adamantane': 1800,
}

# Default frigate armor areas (100m x 20m cylindrical)
FRIGATE_ARMOR_AREAS = {
    'nose': 314,      # pi * r^2
    'lateral': 6280,  # 2 * pi * r * length
    'tail': 314,      # pi * r^2
}


@dataclass
class Ship:
    name: str
    hull_mass_tons: float
    drive: DriveSpec
    reactor: ReactorSpec
    radiator: Radiator
    heatsink: Heatsink
    battery: Battery
    weapons: List[Weapon] = field(default_factory=list)
    armor_zones: List[ArmorZone] = field(default_factory=list)
    crew_misc_tons: float = 80  # Crew, life support, sensors, etc.
    propellant_tons: float = 0  # Calculated or set

    @property
    def weapons_mass_tons(self) -> float:
        return sum(w.mass_tons for w in self.weapons)

    @property
    def ammo_mass_tons(self) -> float:
        return sum(w.total_ammo_tons for w in self.weapons)

    @property
    def armor_mass_tons(self) -> float:
        return sum(a.mass_tons for a in self.armor_zones)

    @property
    def drive_system_mass_tons(self) -> float:
        """Total drive system mass = drive module + reactor"""
        return self.drive.drive_mass_tons + self.reactor.mass_tons

    @property
    def dry_mass_tons(self) -> float:
        """Mass without propellant"""
        return (
            self.hull_mass_tons +
            self.drive.drive_mass_tons +
            self.reactor.mass_tons +
            self.radiator.mass_tons +
            self.heatsink.mass_tons +
            self.battery.mass_tons +
            self.weapons_mass_tons +
            self.ammo_mass_tons +
            self.armor_mass_tons +
            self.crew_misc_tons
        )

    @property
    def wet_mass_tons(self) -> float:
        """Total mass with propellant"""
        return self.dry_mass_tons + self.propellant_tons

    @property
    def wet_mass_kg(self) -> float:
        return self.wet_mass_tons * 1000

    @property
    def dry_mass_kg(self) -> float:
        return self.dry_mass_tons * 1000

    def max_acceleration_g(self, mass_tons: float = None) -> float:
        """Max acceleration in g at given mass (default: wet mass)"""
        if mass_tons is None:
            mass_tons = self.wet_mass_tons
        mass_kg = mass_tons * 1000
        accel_ms2 = self.drive.thrust_n / mass_kg
        return accel_ms2 / G

    def delta_v_kms(self) -> float:
        """Delta-v using Tsiolkovsky equation"""
        if self.propellant_tons <= 0:
            return 0
        mass_ratio = self.wet_mass_tons / self.dry_mass_tons
        return self.drive.exhaust_vel_kms * np.log(mass_ratio)

    def propellant_for_delta_v(self, target_delta_v_kms: float) -> float:
        """Calculate propellant needed for target delta-v"""
        # delta_v = v_e * ln(m_wet / m_dry)
        # m_wet / m_dry = exp(delta_v / v_e)
        # m_wet = m_dry * exp(delta_v / v_e)
        # propellant = m_wet - m_dry
        mass_ratio = np.exp(target_delta_v_kms / self.drive.exhaust_vel_kms)
        wet_mass = self.dry_mass_tons * mass_ratio
        return wet_mass - self.dry_mass_tons

    def propellant_for_accel(self, target_accel_g: float) -> float:
        """Calculate max propellant mass to maintain target acceleration"""
        # accel = thrust / mass
        # mass = thrust / accel
        target_accel_ms2 = target_accel_g * G
        max_mass_kg = self.drive.thrust_n / target_accel_ms2
        max_mass_tons = max_mass_kg / 1000
        return max(0, max_mass_tons - self.dry_mass_tons)


def create_combat_frigate(armor_config: Dict[str, tuple] = None,
                          drive_key: str = 'protium_nova_x6',
                          reactor_key: str = None,
                          heatsink_mass: float = 256) -> Ship:
    """
    Create combat frigate with proper drive-reactor pairing.

    armor_config: dict of {zone: (material, points)}
    drive_key: key from DRIVES dict
    reactor_key: key from REACTORS dict (auto-selected if None)
    heatsink_mass: heatsink mass in tons (can reduce for lighter ship)
    """
    # Default armor config
    if armor_config is None:
        armor_config = {
            'nose': ('adamantane', 10),
            'lateral': ('composite', 3),
            'tail': ('titanium', 2),
        }

    # Build armor zones
    armor_zones = []
    for zone, (material, points) in armor_config.items():
        armor_zones.append(ArmorZone(
            zone=zone,
            material=material,
            points=points,
            area_m2=FRIGATE_ARMOR_AREAS[zone],
            density_kg_m3=ARMOR_MATERIALS[material],
        ))

    # Get drive from database
    drive = DRIVES.get(drive_key, DRIVES['protium_nova_x6'])

    # Auto-select compatible reactor if not specified
    if reactor_key is None:
        reactor = get_best_reactor_for_drive(drive)
    else:
        reactor = REACTORS.get(reactor_key)
        # Verify compatibility
        if reactor.reactor_class != drive.required_reactor and drive.required_reactor != 'any':
            raise ValueError(f"Reactor {reactor.name} ({reactor.reactor_class}) incompatible with "
                           f"drive {drive.name} (requires {drive.required_reactor})")

    # Weapons loadout
    weapons = [
        Weapon("Coilgun Battery Mk3", 80, ammo_mass_kg=20, magazine_size=1800),
        Weapon("Coilgun Battery Mk3", 80, ammo_mass_kg=20, magazine_size=1800),
        Weapon("Spinal Coilgun Mk3", 200, ammo_mass_kg=100, magazine_size=450),
        Weapon("PD Arc Laser Turret", 20),
        Weapon("PD Arc Laser Turret", 20),
        Weapon("Cobra Launcher", 25, ammo_mass_kg=1600, magazine_size=16),
        Weapon("Cobra Launcher", 25, ammo_mass_kg=1600, magazine_size=16),
    ]

    # Calculate heatsink capacity based on mass (4.1 GJ/ton for lithium)
    heatsink_capacity = heatsink_mass * 4.1

    ship = Ship(
        name=f"Frigate ({drive.name})",
        hull_mass_tons=150,
        drive=drive,
        reactor=reactor,
        radiator=Radiator("Lithium Spray", specific_power_kw_kg=13, mass_tons=10),
        heatsink=Heatsink("Heavy Lithium", capacity_gj=heatsink_capacity, mass_tons=heatsink_mass),
        battery=Battery("Superconducting Coil", capacity_gj=160, mass_tons=20),
        weapons=weapons,
        armor_zones=armor_zones,
        crew_misc_tons=80,
    )

    return ship


def print_ship_breakdown(ship: Ship):
    """Print detailed mass breakdown"""
    print(f"\n{'='*60}")
    print(f"SHIP: {ship.name}")
    print(f"{'='*60}")

    print(f"\n--- MASS BREAKDOWN ---")
    print(f"{'Component':<35} {'Mass (tons)':>12}")
    print(f"{'-'*47}")
    print(f"{'Hull Structure':<35} {ship.hull_mass_tons:>12.1f}")
    print(f"{'Drive (' + ship.drive.name[:20] + ')':<35} {ship.drive.drive_mass_tons:>12.1f}")
    print(f"{'Reactor (' + ship.reactor.name[:18] + ')':<35} {ship.reactor.mass_tons:>12.1f}")
    print(f"{'Radiator':<35} {ship.radiator.mass_tons:>12.1f}")
    print(f"{'Heatsink':<35} {ship.heatsink.mass_tons:>12.1f}")
    print(f"{'Battery':<35} {ship.battery.mass_tons:>12.1f}")
    print(f"{'Crew/Misc':<35} {ship.crew_misc_tons:>12.1f}")

    print(f"\n{'--- WEAPONS ---':<30}")
    for w in ship.weapons:
        print(f"  {w.name:<28} {w.mass_tons:>12.1f}")
    print(f"{'Weapons Subtotal':<30} {ship.weapons_mass_tons:>12.1f}")
    print(f"{'Ammunition':<30} {ship.ammo_mass_tons:>12.1f}")

    print(f"\n{'--- ARMOR ---':<30}")
    for a in ship.armor_zones:
        desc = f"  {a.zone.capitalize()} ({a.points}pt {a.material})"
        print(f"{desc:<30} {a.mass_tons:>12.1f}")
    print(f"{'Armor Subtotal':<30} {ship.armor_mass_tons:>12.1f}")

    print(f"\n{'-'*42}")
    print(f"{'DRY MASS':<30} {ship.dry_mass_tons:>12.1f}")
    print(f"{'Propellant':<30} {ship.propellant_tons:>12.1f}")
    print(f"{'WET MASS':<30} {ship.wet_mass_tons:>12.1f}")


def compare_drive_reactor_combos():
    """Compare different drive+reactor combinations for combat"""

    print("\n" + "="*90)
    print("DRIVE + REACTOR COMBINATIONS FOR COMBAT")
    print("="*90)

    # Test each drive with its best reactor
    drive_keys = ['protium_nova_x6', 'protium_nova_x4', 'borane_plasmajet_x6',
                  'protium_converter_x4', 'helion_torus_x6']

    print(f"\n{'Drive':<28} {'Reactor':<25} {'Thrust':>8} {'EV':>7} {'Sys Mass':>10} {'Max@4g':>10}")
    print(f"{'':28} {'':25} {'(MN)':>8} {'(km/s)':>7} {'(tons)':>10} {'(tons)':>10}")
    print("-" * 95)

    for dk in drive_keys:
        drive = DRIVES[dk]
        reactor = get_best_reactor_for_drive(drive)
        sys_mass = drive.drive_mass_tons + reactor.mass_tons
        max_mass_4g = drive.thrust_n / (TARGET_ACCEL_G * G) / 1000

        print(f"{drive.name:<28} {reactor.name:<25} {drive.thrust_mn:>8.1f} "
              f"{drive.exhaust_vel_kms:>7.0f} {sys_mass:>10.0f} {max_mass_4g:>10.0f}")


def analyze_armor_vs_acceleration(drive_key: str = 'protium_nova_x6'):
    """Main analysis: how armor affects 4g requirement"""

    drive = DRIVES[drive_key]
    reactor = get_best_reactor_for_drive(drive)

    print("\n" + "="*70)
    print(f"ARMOR vs 4G ANALYSIS - {drive.name}")
    print(f"Reactor: {reactor.name} ({reactor.mass_tons:.0f}t)")
    print("="*70)

    # Create baseline ship with NO armor
    no_armor = {'nose': ('adamantane', 0), 'lateral': ('composite', 0), 'tail': ('titanium', 0)}
    baseline = create_combat_frigate(no_armor, drive_key)

    # Calculate max propellant for 4g with no armor
    max_prop_no_armor = baseline.propellant_for_accel(TARGET_ACCEL_G)
    baseline.propellant_tons = max(0, max_prop_no_armor)

    print(f"\n--- BASELINE (No Armor) ---")
    print(f"Drive system mass: {baseline.drive_system_mass_tons:.1f} tons")
    print(f"Dry mass: {baseline.dry_mass_tons:.1f} tons")
    print(f"Max propellant for 4g: {max_prop_no_armor:.1f} tons")
    if max_prop_no_armor > 0:
        print(f"Wet mass at 4g: {baseline.wet_mass_tons:.1f} tons")
        print(f"Delta-v: {baseline.delta_v_kms():.1f} km/s")
    print(f"Acceleration (dry): {baseline.max_acceleration_g(baseline.dry_mass_tons):.2f} g")

    # Now test different armor configurations
    print(f"\n{'='*90}")
    print("ARMOR CONFIGURATION COMPARISON")
    print(f"{'='*90}")
    print(f"\n{'Config':<35} {'Armor':>8} {'Dry':>8} {'Prop':>8} {'Wet':>8} {'DV':>8} {'Accel':>6}")
    print(f"{'':35} {'(tons)':>8} {'(tons)':>8} {'(tons)':>8} {'(tons)':>8} {'(km/s)':>8} {'(g)':>6}")
    print("-" * 90)

    configs = [
        ("No armor", {'nose': ('adamantane', 0), 'lateral': ('composite', 0), 'tail': ('titanium', 0)}),
        ("Light (3/1/1 Composite)", {'nose': ('composite', 3), 'lateral': ('composite', 1), 'tail': ('composite', 1)}),
        ("Medium (5/2/2 Composite)", {'nose': ('composite', 5), 'lateral': ('composite', 2), 'tail': ('composite', 2)}),
        ("Reference (10/3/2 Mixed)", {'nose': ('adamantane', 10), 'lateral': ('composite', 3), 'tail': ('titanium', 2)}),
        ("Heavy (15/5/3 Adamantane)", {'nose': ('adamantane', 15), 'lateral': ('adamantane', 5), 'tail': ('adamantane', 3)}),
        ("Assault (20/8/5 Adamantane)", {'nose': ('adamantane', 20), 'lateral': ('adamantane', 8), 'tail': ('adamantane', 5)}),
    ]

    for name, armor_cfg in configs:
        ship = create_combat_frigate(armor_cfg, drive_key)
        max_prop = ship.propellant_for_accel(TARGET_ACCEL_G)

        if max_prop > 0:
            ship.propellant_tons = max_prop
            dv = ship.delta_v_kms()
            accel = ship.max_acceleration_g()
            print(f"{name:<35} {ship.armor_mass_tons:>8.1f} {ship.dry_mass_tons:>8.1f} "
                  f"{ship.propellant_tons:>8.1f} {ship.wet_mass_tons:>8.1f} {dv:>8.1f} {accel:>6.2f}")
        else:
            # Ship too heavy for 4g even with no propellant!
            ship.propellant_tons = 0
            accel = ship.max_acceleration_g(ship.dry_mass_tons)
            print(f"{name:<35} {ship.armor_mass_tons:>8.1f} {ship.dry_mass_tons:>8.1f} "
                  f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {accel:>6.2f}*")

    print("\n* = Cannot achieve 4g, shown acceleration is at dry mass (no propellant)")


def find_armor_limit(drive_key: str = 'protium_nova_x6'):
    """Find the exact armor point where 4g becomes impossible"""

    drive = DRIVES[drive_key]
    reactor = get_best_reactor_for_drive(drive)

    print("\n" + "="*70)
    print(f"FINDING ARMOR LIMIT FOR 4G - {drive.name}")
    print("="*70)

    # Test nose armor scaling (most mass-efficient location)
    print(f"\n--- Nose Armor Only (Adamantane) ---")
    print(f"{'Points':<8} {'Armor Mass':>12} {'Dry Mass':>12} {'Max Prop':>12} {'Delta-V':>12} {'Verdict':>15}")
    print("-" * 75)

    for points in range(0, 201, 10):
        ship = create_combat_frigate({
            'nose': ('adamantane', points),
            'lateral': ('composite', 0),
            'tail': ('titanium', 0)
        }, drive_key)
        max_prop = ship.propellant_for_accel(TARGET_ACCEL_G)

        if max_prop > 0:
            ship.propellant_tons = max_prop
            dv = ship.delta_v_kms()
            verdict = "OK" if dv >= 300 else f"Low DV ({dv:.0f})"
        else:
            dv = 0
            verdict = "TOO HEAVY!"

        print(f"{points:<8} {ship.armor_mass_tons:>12.1f} {ship.dry_mass_tons:>12.1f} "
              f"{max_prop:>12.1f} {dv:>12.1f} {verdict:>15}")

        if max_prop <= 0:
            break

    # Calculate exact limit with binary search
    print(f"\n--- Calculating Exact Limit ---")

    low, high = 0, 500
    while high - low > 1:
        mid = (low + high) / 2
        ship = create_combat_frigate({
            'nose': ('adamantane', mid),
            'lateral': ('composite', 0),
            'tail': ('titanium', 0)
        }, drive_key)
        if ship.propellant_for_accel(TARGET_ACCEL_G) > 0:
            low = mid
        else:
            high = mid

    max_nose_points = low
    ship = create_combat_frigate({
        'nose': ('adamantane', max_nose_points),
        'lateral': ('composite', 0),
        'tail': ('titanium', 0)
    }, drive_key)

    print(f"\nMax nose armor for 4g: {max_nose_points:.0f} points ({max_nose_points:.0f} cm)")
    print(f"Armor mass at limit: {ship.armor_mass_tons:.1f} tons")
    print(f"Dry mass at limit: {ship.dry_mass_tons:.1f} tons")

    # What's the max total mass for 4g?
    max_mass_kg = ship.drive.thrust_n / (TARGET_ACCEL_G * G)
    max_mass_tons = max_mass_kg / 1000
    print(f"\nDrive thrust: {ship.drive.thrust_mn} MN")
    print(f"Max total mass for 4g: {max_mass_tons:.1f} tons")


def print_final_recommendation(drive_key: str = 'borane_plasmajet_x6'):
    """Print recommended configuration"""

    print("\n" + "="*70)
    print("RECOMMENDED CONFIGURATION")
    print("="*70)

    # Build recommended ship with lighter heatsink for combat
    ship = create_combat_frigate(
        armor_config={
            'nose': ('adamantane', 10),
            'lateral': ('composite', 3),
            'tail': ('titanium', 2)
        },
        drive_key=drive_key,
        heatsink_mass=100  # Lighter heatsink for combat ship
    )

    # Set propellant for 350 km/s delta-v
    target_dv = 350
    ship.propellant_tons = ship.propellant_for_delta_v(target_dv)

    print_ship_breakdown(ship)

    print(f"\n--- PERFORMANCE ---")
    print(f"Drive: {ship.drive.name}")
    print(f"Reactor: {ship.reactor.name}")
    print(f"Drive thrust: {ship.drive.thrust_mn} MN")
    print(f"Exhaust velocity: {ship.drive.exhaust_vel_kms} km/s")
    print(f"Delta-V: {ship.delta_v_kms():.1f} km/s")
    print(f"Max accel (wet): {ship.max_acceleration_g():.2f} g")
    print(f"Max accel (dry): {ship.max_acceleration_g(ship.dry_mass_tons):.2f} g")

    # Check if we meet 4g requirement
    if ship.max_acceleration_g() >= 4.0:
        print(f"\n[OK] Ship meets 4g combat acceleration requirement!")
    else:
        deficit = ship.wet_mass_tons - ship.drive.thrust_n/(4*G)/1000
        print(f"\n[WARNING] Ship does NOT meet 4g requirement!")
        print(f"  Need to reduce mass by {deficit:.1f} tons")
        print(f"  Or increase thrust by {deficit * 4 * G / 1e6:.1f} MN")


if __name__ == "__main__":
    # Compare all drive+reactor options
    compare_drive_reactor_combos()

    # Analyze best option: Borane Plasmajet (good balance of thrust + Isp)
    analyze_armor_vs_acceleration('borane_plasmajet_x6')
    find_armor_limit('borane_plasmajet_x6')

    # Also test Protium Nova (more thrust but heavier reactor)
    analyze_armor_vs_acceleration('protium_nova_x6')

    # Final recommendation
    print_final_recommendation('borane_plasmajet_x6')
