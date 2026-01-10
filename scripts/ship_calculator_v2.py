#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Ship Mass/Delta-V/Acceleration Calculator v2 for AI Commanders

Uses VERIFIED Terra Invicta data extracted from game files.
Target: 3g combat acceleration (more realistic than 4g)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

G = 9.81  # m/s²
TARGET_ACCEL_G = 3.0  # 3g target (reduced from 4g)


# =============================================================================
# VERIFIED TERRA INVICTA DATA (from game file extraction)
# =============================================================================

@dataclass
class DriveSpec:
    name: str
    thrust_mn: float
    exhaust_vel_kms: float
    drive_mass_tons: float  # From specificPower calc or flatMass
    efficiency: float
    required_reactor_class: str

    @property
    def thrust_n(self) -> float:
        return self.thrust_mn * 1e6


@dataclass
class ReactorSpec:
    name: str
    output_gw: float
    mass_tons: float
    efficiency: float
    reactor_class: str
    requires_exotics: bool = False


# Verified drives sorted by thrust * Isp (combat effectiveness)
DRIVES = {
    # === INERTIAL CONFINEMENT FUSION (best high-thrust + high-Isp) ===
    'protium_nova_x6': DriveSpec("Protium Nova Torch x6", 39.6, 1000, 0, 0.97, 'icf'),
    'protium_nova_x5': DriveSpec("Protium Nova Torch x5", 33.0, 1000, 0, 0.97, 'icf'),
    'protium_nova_x4': DriveSpec("Protium Nova Torch x4", 26.4, 1000, 0, 0.97, 'icf'),
    'protium_converter_x6': DriveSpec("Protium Converter Torch x6", 58.56, 10256, 0, 0.98, 'icf'),
    'protium_converter_x4': DriveSpec("Protium Converter Torch x4", 39.04, 10256, 0, 0.98, 'icf'),
    'borane_nova_x6': DriveSpec("Borane Nova Lantern x6", 16.02, 678, 0, 0.99, 'icf'),

    # === HYBRID CONFINEMENT FUSION ===
    'borane_plasmajet_x6': DriveSpec("Borane Plasmajet Torch x6", 30.24, 714, 0, 0.95, 'hybrid'),
    'borane_plasmajet_x5': DriveSpec("Borane Plasmajet Torch x5", 25.2, 714, 0, 0.95, 'hybrid'),
    'borane_plasmajet_x4': DriveSpec("Borane Plasmajet Torch x4", 20.16, 714, 0, 0.95, 'hybrid'),

    # === GAS CORE FISSION (high thrust, lower Isp) ===
    'lodestar_x6': DriveSpec("Lodestar Fission Lantern x6", 66.0, 31.4, 0, 0.925, 'gas_core'),
    'lodestar_x5': DriveSpec("Lodestar Fission Lantern x5", 55.0, 31.4, 0, 0.925, 'gas_core'),
    'lodestar_x4': DriveSpec("Lodestar Fission Lantern x4", 44.0, 31.4, 0, 0.925, 'gas_core'),
    'firestar_x6': DriveSpec("Firestar Fission Lantern x6", 30.0, 50, 0, 0.85, 'gas_core'),

    # === TOKAMAK (lower thrust) ===
    'protium_torus_x6': DriveSpec("Protium Torus Lantern x6", 10.1, 952, 0, 0.95, 'tokamak'),
    'helion_torus_x6': DriveSpec("Helion Torus Lantern x6", 3.34, 690, 0, 0.925, 'tokamak'),

    # === NUCLEAR SALT WATER (massive but powerful) ===
    'neutron_flux_x6': DriveSpec("Neutron Flux Torch x6", 78.0, 1700, 9613.5, 0.80, 'any'),
    'neutron_flux_x5': DriveSpec("Neutron Flux Torch x5", 65.0, 1700, 8011.2, 0.80, 'any'),
}

# Verified reactors - INCLUDING exotic ones for top tier
REACTORS = {
    # === INERTIAL CONFINEMENT FUSION ===
    'icf_vii': ReactorSpec("ICF VII", 306430, 612.86, 0.999, 'icf', True),  # EXOTIC
    'icf_vi': ReactorSpec("ICF VI", 20420, 1388.56, 0.99, 'icf', True),   # EXOTIC
    'icf_v': ReactorSpec("ICF V", 19090, 4772.5, 0.975, 'icf', True),     # EXOTIC
    'icf_iv': ReactorSpec("ICF IV", 5500, 2750, 0.95, 'icf', False),
    'icf_iii': ReactorSpec("ICF III", 3170, 3170, 0.92, 'icf', False),
    'icf_ii': ReactorSpec("ICF II", 860, 1720, 0.89, 'icf', False),

    # === HYBRID CONFINEMENT FUSION ===
    'hybrid_iv': ReactorSpec("Hybrid IV", 11370, 568.5, 0.99, 'hybrid', False),
    'hybrid_iii': ReactorSpec("Hybrid III", 1900, 950, 0.99, 'hybrid', False),
    'hybrid_ii': ReactorSpec("Hybrid II", 510, 510, 0.98, 'hybrid', False),

    # === GAS CORE FISSION ===
    'gas_vi': ReactorSpec("Gas Core VI", 1650, 1650, 0.96, 'gas_core', False),
    'gas_v': ReactorSpec("Gas Core V", 1650, 5775, 0.95, 'gas_core', False),
    'gas_iv': ReactorSpec("Gas Core IV", 1650, 16500, 0.93, 'gas_core', False),
    'gas_iii': ReactorSpec("Gas Core III", 150, 450, 0.95, 'gas_core', False),

    # === TOKAMAK ===
    'tokamak_v': ReactorSpec("Tokamak V", 5060, 506, 0.99, 'tokamak', True),  # EXOTIC
    'tokamak_iv': ReactorSpec("Tokamak IV", 1260, 630, 0.985, 'tokamak', False),
    'tokamak_iii': ReactorSpec("Tokamak III", 624, 624, 0.96, 'tokamak', False),

    # === Z-PINCH (for any) ===
    'zpinch_iv': ReactorSpec("Z-Pinch IV", 3970, 1588, 0.98, 'zpinch', False),
}


# Verified hull data
HULLS = {
    'fighter': {'name': 'STO Fighter', 'mass': 30, 'si': 3, 'crew': 2},
    'gunship': {'name': 'Gunship', 'mass': 178, 'si': 4, 'crew': 3},
    'escort': {'name': 'Escort', 'mass': 350, 'si': 7, 'crew': 4},
    'corvette': {'name': 'Corvette', 'mass': 400, 'si': 8, 'crew': 8},
    'frigate': {'name': 'Frigate', 'mass': 600, 'si': 12, 'crew': 20},
    'destroyer': {'name': 'Destroyer', 'mass': 825, 'si': 18, 'crew': 40},
    'cruiser': {'name': 'Cruiser', 'mass': 1000, 'si': 20, 'crew': 60},
    'battlecruiser': {'name': 'Battlecruiser', 'mass': 1200, 'si': 24, 'crew': 70},
    'battleship': {'name': 'Battleship', 'mass': 1600, 'si': 40, 'crew': 80},
}

# Verified weapons (mass in tons)
WEAPONS = {
    'coilgun_battery_mk3': {'name': 'Coilgun Battery Mk3', 'mass': 80, 'ammo_kg': 20, 'mag': 1800},
    'heavy_coilgun_mk3': {'name': 'Heavy Coilgun Battery Mk3', 'mass': 160, 'ammo_kg': 40, 'mag': 2160},
    'spinal_coiler_mk3': {'name': 'Spinal Coiler Mk3', 'mass': 200, 'ammo_kg': 100, 'mag': 450},
    'light_coilgun_mk3': {'name': 'Light Coilgun Battery Mk3', 'mass': 40, 'ammo_kg': 10, 'mag': 1500},
    'pd_laser': {'name': 'PD Laser Turret', 'mass': 20, 'ammo_kg': 0, 'mag': 0},
    'torpedo_launcher': {'name': 'Torpedo Launcher', 'mass': 25, 'ammo_kg': 1600, 'mag': 16},
}

# Verified thermal systems
HEATSINKS = {
    'lithium': {'name': 'Lithium Heat Sink', 'capacity_gj': 525, 'mass': 128},
    'heavy_lithium': {'name': 'Heavy Lithium Heat Sink', 'capacity_gj': 1050, 'mass': 256},
    'sodium': {'name': 'Sodium Heat Sink', 'capacity_gj': 370, 'mass': 230},
    'exotic': {'name': 'Exotic Heat Sink', 'capacity_gj': 1800, 'mass': 250},
}

BATTERIES = {
    'superconducting': {'name': 'Superconducting Coil', 'capacity_gj': 160, 'mass': 20},
    'quantum': {'name': 'Quantum Battery', 'capacity_gj': 80, 'mass': 26},
}


def get_compatible_reactors(drive: DriveSpec) -> List[ReactorSpec]:
    """Get reactors compatible with a drive."""
    if drive.required_reactor_class == 'any':
        return list(REACTORS.values())
    return [r for r in REACTORS.values() if r.reactor_class == drive.required_reactor_class]


def get_best_reactor(drive: DriveSpec, allow_exotics: bool = True) -> ReactorSpec:
    """Get lightest compatible reactor."""
    compatible = get_compatible_reactors(drive)
    if not allow_exotics:
        compatible = [r for r in compatible if not r.requires_exotics]
    if not compatible:
        raise ValueError(f"No compatible reactor for {drive.name}")
    return min(compatible, key=lambda r: r.mass_tons)


def analyze_drive_reactor_combos():
    """Show all drive+reactor combinations with 3g capability."""

    print("=" * 100)
    print(f"DRIVE + REACTOR COMBINATIONS FOR {TARGET_ACCEL_G}g COMBAT")
    print("=" * 100)

    results = []
    seen = set()  # Track (drive, reactor) pairs to avoid duplicates

    for dk, drive in DRIVES.items():
        for allow_ex in [True, False]:
            try:
                reactor = get_best_reactor(drive, allow_exotics=allow_ex)
            except ValueError:
                continue

            # Skip duplicate combinations
            key = (drive.name, reactor.name)
            if key in seen:
                continue
            seen.add(key)

            sys_mass = drive.drive_mass_tons + reactor.mass_tons
            max_mass_3g = drive.thrust_n / (TARGET_ACCEL_G * G) / 1000
            available = max_mass_3g - sys_mass

            results.append({
                'drive': drive.name,
                'reactor': reactor.name,
                'thrust': drive.thrust_mn,
                'ev': drive.exhaust_vel_kms,
                'sys_mass': sys_mass,
                'max_mass': max_mass_3g,
                'available': available,
                'exotic': reactor.requires_exotics,
            })

    # Sort by available mass
    results.sort(key=lambda x: -x['available'])

    print(f"\n{'Drive':<30} {'Reactor':<15} {'Thrust':>8} {'EV':>7} {'SysMass':>8} {'Max@3g':>8} {'Avail':>8} {'Ex':<4}")
    print(f"{'':30} {'':15} {'(MN)':>8} {'(km/s)':>7} {'(t)':>8} {'(t)':>8} {'(t)':>8} {'':<4}")
    print("-" * 105)

    for r in results:
        exotic = "YES" if r['exotic'] else "no"
        avail = f"{r['available']:.0f}" if r['available'] > 0 else f"{r['available']:.0f}!"
        print(f"{r['drive']:<30} {r['reactor']:<15} {r['thrust']:>8.1f} {r['ev']:>7.0f} "
              f"{r['sys_mass']:>8.0f} {r['max_mass']:>8.0f} {avail:>8} {exotic:<4}")


def calculate_ship_mass(hull_key: str, weapons: List[str], heatsink_key: str = 'lithium',
                        battery_key: str = 'superconducting', armor_tons: float = 100) -> dict:
    """Calculate total ship mass from components."""

    hull = HULLS[hull_key]
    hs = HEATSINKS[heatsink_key]
    bat = BATTERIES[battery_key]

    weapons_mass = sum(WEAPONS[w]['mass'] for w in weapons)
    ammo_mass = sum(WEAPONS[w]['ammo_kg'] * WEAPONS[w]['mag'] / 1000 for w in weapons)

    # Radiator (10 tons for Lithium Spray)
    radiator_mass = 10

    # Crew/misc (estimated at 2 tons per crew + 20 tons misc)
    crew_misc = hull['crew'] * 2 + 20

    dry_mass = (hull['mass'] + weapons_mass + ammo_mass + hs['mass'] +
                bat['mass'] + radiator_mass + crew_misc + armor_tons)

    return {
        'hull': hull['mass'],
        'weapons': weapons_mass,
        'ammo': ammo_mass,
        'heatsink': hs['mass'],
        'battery': bat['mass'],
        'radiator': radiator_mass,
        'crew_misc': crew_misc,
        'armor': armor_tons,
        'dry_total': dry_mass,
    }


def find_viable_configs():
    """Find ship configurations that can achieve 3g."""

    print("\n" + "=" * 100)
    print("VIABLE 3g COMBAT CONFIGURATIONS")
    print("=" * 100)

    # Test different weapon loadouts - sized for different hulls
    loadouts = {
        'minimal': ['light_coilgun_mk3', 'pd_laser'],  # For gunships/small craft
        'light': ['light_coilgun_mk3', 'light_coilgun_mk3', 'pd_laser', 'pd_laser'],
        'medium': ['coilgun_battery_mk3', 'coilgun_battery_mk3', 'pd_laser', 'pd_laser'],
        'heavy': ['coilgun_battery_mk3', 'coilgun_battery_mk3', 'spinal_coiler_mk3', 'pd_laser', 'pd_laser'],
    }

    # Include ALL promising drives including the exotic ICF options
    best_drives = ['protium_nova_x6', 'protium_converter_x6', 'borane_plasmajet_x6', 'lodestar_x6', 'protium_converter_x4']

    for drive_key in best_drives:
        drive = DRIVES[drive_key]
        print(f"\n--- {drive.name} (Thrust: {drive.thrust_mn} MN, EV: {drive.exhaust_vel_kms} km/s) ---")

        seen_reactors = set()
        for allow_ex in [True, False]:
            try:
                reactor = get_best_reactor(drive, allow_exotics=allow_ex)
            except ValueError:
                continue

            if reactor.name in seen_reactors:
                continue
            seen_reactors.add(reactor.name)

            sys_mass = drive.drive_mass_tons + reactor.mass_tons
            max_mass = drive.thrust_n / (TARGET_ACCEL_G * G) / 1000

            ex_str = "(EXOTIC)" if reactor.requires_exotics else "(non-exotic)"
            print(f"\n  Reactor: {reactor.name} ({reactor.mass_tons:.0f}t) {ex_str}")
            print(f"  Drive system: {sys_mass:.0f}t | Max mass @ 3g: {max_mass:.0f}t")
            print(f"  {'Loadout':<12} {'Hull':<12} {'ShipMass':>10} {'Total':>10} {'Margin':>10} {'DeltaV':>10}")
            print(f"  {'-'*70}")

            for loadout_name, weapons in loadouts.items():
                for hull_key in ['gunship', 'escort', 'corvette', 'frigate', 'destroyer']:
                    ship = calculate_ship_mass(hull_key, weapons, armor_tons=50)
                    total = ship['dry_total'] + sys_mass
                    margin = max_mass - total

                    # Calculate delta-v if viable
                    if margin > 0:
                        # Propellant = margin (all remaining mass budget)
                        prop = margin * 0.8  # Leave some margin
                        wet = total + prop
                        dry = total
                        dv = drive.exhaust_vel_kms * np.log(wet / dry)
                        dv_str = f"{dv:.0f} km/s"
                    else:
                        dv_str = "N/A"

                    status = "OK" if margin > 100 else ("tight" if margin > 0 else "FAIL")
                    print(f"  {loadout_name:<12} {hull_key:<12} {ship['dry_total']:>10.0f} "
                          f"{total:>10.0f} {margin:>10.0f} {dv_str:>10} {status}")


def recommend_combat_ship():
    """Recommend optimal combat ship configuration."""

    print("\n" + "=" * 100)
    print("RECOMMENDED COMBAT CONFIGURATION")
    print("=" * 100)

    # Best exotic option: Protium Nova x6 + ICF VII
    # Best non-exotic: Borane Plasmajet x6 + Hybrid IV

    configs = [
        ('protium_converter_x6', 'icf_vii', True, 'EXOTIC - Interplanetary Combat Ship'),
        ('lodestar_x6', 'gas_iii', False, 'NON-EXOTIC - Short Range Gunboat'),
    ]

    for drive_key, reactor_key, is_exotic, label in configs:
        drive = DRIVES[drive_key]
        reactor = REACTORS[reactor_key]

        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

        # Choose appropriate hull for each config
        if is_exotic:
            # Exotic: can field corvettes with good loadout
            hull_key = 'corvette'
            weapons = ['coilgun_battery_mk3', 'coilgun_battery_mk3', 'pd_laser', 'pd_laser']
        else:
            # Non-exotic: larger ships work, frigate with medium loadout
            hull_key = 'frigate'
            weapons = ['coilgun_battery_mk3', 'coilgun_battery_mk3', 'pd_laser', 'pd_laser']

        ship = calculate_ship_mass(hull_key, weapons, heatsink_key='lithium', armor_tons=50)

        sys_mass = drive.drive_mass_tons + reactor.mass_tons
        dry_total = ship['dry_total'] + sys_mass
        max_mass = drive.thrust_n / (TARGET_ACCEL_G * G) / 1000
        prop_budget = max_mass - dry_total

        print(f"\nDrive: {drive.name}")
        print(f"  Thrust: {drive.thrust_mn} MN")
        print(f"  Exhaust velocity: {drive.exhaust_vel_kms} km/s")
        print(f"  Drive mass: {drive.drive_mass_tons:.0f} tons")
        print(f"\nReactor: {reactor.name}")
        print(f"  Output: {reactor.output_gw:.0f} GW")
        print(f"  Mass: {reactor.mass_tons:.0f} tons")
        print(f"  Requires exotics: {'Yes' if is_exotic else 'No'}")

        hull_name = HULLS[hull_key]['name']
        print(f"\nShip Mass Breakdown:")
        print(f"  Hull ({hull_name}):     {ship['hull']:>6.0f} t")
        print(f"  Weapons:            {ship['weapons']:>6.0f} t")
        print(f"  Ammunition:         {ship['ammo']:>6.0f} t")
        print(f"  Heatsink:           {ship['heatsink']:>6.0f} t")
        print(f"  Battery:            {ship['battery']:>6.0f} t")
        print(f"  Radiator:           {ship['radiator']:>6.0f} t")
        print(f"  Crew/Misc:          {ship['crew_misc']:>6.0f} t")
        print(f"  Armor:              {ship['armor']:>6.0f} t")
        print(f"  Drive System:       {sys_mass:>6.0f} t")
        print(f"  {'─'*25}")
        print(f"  DRY TOTAL:          {dry_total:>6.0f} t")

        print(f"\nPerformance @ {TARGET_ACCEL_G}g:")
        print(f"  Max ship mass:      {max_mass:>6.0f} t")
        print(f"  Propellant budget:  {prop_budget:>6.0f} t")

        if prop_budget > 0:
            wet = dry_total + prop_budget
            dv = drive.exhaust_vel_kms * np.log(wet / dry_total)
            print(f"  Wet mass:           {wet:>6.0f} t")
            print(f"  Delta-V:            {dv:>6.0f} km/s")
            print(f"\n  [OK] Configuration achieves {TARGET_ACCEL_G}g!")
        else:
            accel = drive.thrust_n / (dry_total * 1000) / G
            print(f"\n  [FAIL] Too heavy! Max acceleration: {accel:.2f}g")
            print(f"  Need to reduce mass by {-prop_budget:.0f} tons")


if __name__ == "__main__":
    analyze_drive_reactor_combos()
    find_viable_configs()
    recommend_combat_ship()
