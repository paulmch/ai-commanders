#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Armor Budget Calculator for AI Commanders

Given the exotic Protium Converter x6 + ICF VII configuration,
calculate how much armor we can add while maintaining 3g combat
acceleration and 1000 km/s delta-v.
"""

import json
import numpy as np
from pathlib import Path

# Constants
G = 9.81  # m/s²
TARGET_ACCEL_G = 3.0
TARGET_DELTA_V_KMS = 1000  # Reduced from 2700 for more armor

# Protium Converter x6 + ICF VII specs
THRUST_MN = 58.56
EXHAUST_VEL_KMS = 10256
REACTOR_MASS_T = 613  # ICF VII

# Hull data with hardpoints
HULLS = {
    'gunship':      {'mass': 178, 'nose_hp': 1, 'hull_hp': 0, 'utility': 2, 'crew': 3},
    'escort':       {'mass': 350, 'nose_hp': 0, 'hull_hp': 2, 'utility': 2, 'crew': 4},
    'corvette':     {'mass': 400, 'nose_hp': 1, 'hull_hp': 1, 'utility': 3, 'crew': 8},
    'frigate':      {'mass': 600, 'nose_hp': 1, 'hull_hp': 2, 'utility': 5, 'crew': 20},
    'destroyer':    {'mass': 825, 'nose_hp': 2, 'hull_hp': 2, 'utility': 5, 'crew': 40},
    'cruiser':      {'mass': 1000, 'nose_hp': 2, 'hull_hp': 3, 'utility': 7, 'crew': 60},
}

# Weapon masses (tons)
WEAPONS = {
    'light_coilgun': 40,      # Light Coilgun Battery Mk3
    'coilgun': 80,            # Coilgun Battery Mk3
    'heavy_coilgun': 160,     # Heavy Coilgun Battery Mk3
    'pd_laser': 20,           # PD Laser Turret
    'spinal_coiler': 200,     # Spinal Coiler (nose only)
}

# Ammo per weapon (tons) - approximate
AMMO = {
    'light_coilgun': 15,
    'coilgun': 36,
    'heavy_coilgun': 86,
    'pd_laser': 0,
    'spinal_coiler': 45,
}

def calculate_ship_mass(hull_key, weapons_list):
    """Calculate ship mass with proper hardpoint constraints."""
    hull = HULLS[hull_key]

    weapons_mass = sum(WEAPONS[w] for w in weapons_list)
    ammo_mass = sum(AMMO[w] for w in weapons_list)

    return {
        'hull': hull['mass'],
        'weapons': weapons_mass,
        'ammunition': ammo_mass,
        'heatsink': 128,       # Lithium Heat Sink
        'battery': 20,         # Superconducting
        'radiator': 10,        # Lithium Spray
        'crew_misc': hull['crew'] * 2 + 20,
        'reactor': REACTOR_MASS_T,
    }

# Destroyer with 4 weapons (max hardpoints: 2 nose + 2 hull)
# Balanced loadout: 2x Coilgun Battery + 2x PD Laser
SELECTED_HULL = 'destroyer'
SELECTED_WEAPONS = ['coilgun', 'coilgun', 'pd_laser', 'pd_laser']  # 4 hardpoints

BASE_SHIP_MASS = calculate_ship_mass(SELECTED_HULL, SELECTED_WEAPONS)

def parse_float(s):
    """Parse float, handling comma-formatted numbers."""
    if isinstance(s, (int, float)):
        return float(s)
    return float(str(s).replace(',', ''))


def extract_armor_data():
    """Extract armor types from Terra Invicta game files."""
    ti_path = Path("/mnt/d/SteamLibrary/steamapps/common/Terra Invicta")
    armor_file = ti_path / "TerraInvicta_Data/StreamingAssets/Templates/TIShipArmorTemplate.json"

    with open(armor_file) as f:
        data = json.load(f)

    armors = []
    for armor in data:
        name = armor.get('friendlyName', armor.get('dataName', 'Unknown'))

        # Skip alien armors for human ships
        if 'Alien' in name:
            continue

        armors.append({
            'name': name,
            'density': parse_float(armor.get('density_kgm3', 0)),
            'baryonic_half_cm': parse_float(armor.get('baryonicHalfValue_cm', 0)),
            'xray_half_cm': parse_float(armor.get('xRayHalfValue_cm', 0)),
            'heat_vaporization': parse_float(armor.get('heatofVaporization_MJkg', 0)),
        })

    return armors


def calculate_armor_budget():
    """Calculate how much armor mass we can add."""

    # Calculate base dry mass (no armor)
    base_dry = sum(BASE_SHIP_MASS.values())
    print(f"Base dry mass (no armor): {base_dry} tons")

    # Max wet mass at 3g
    thrust_n = THRUST_MN * 1e6
    max_wet_mass = thrust_n / (TARGET_ACCEL_G * G) / 1000
    print(f"Max wet mass @ {TARGET_ACCEL_G}g: {max_wet_mass:.0f} tons")

    # For target delta-v, what's the mass ratio?
    # dv = Ve * ln(wet/dry)
    # wet/dry = e^(dv/Ve)
    mass_ratio = np.exp(TARGET_DELTA_V_KMS / EXHAUST_VEL_KMS)
    print(f"\nTarget delta-v: {TARGET_DELTA_V_KMS} km/s")
    print(f"Required mass ratio (wet/dry): {mass_ratio:.4f}")

    # Max dry mass = max_wet / mass_ratio
    max_dry_mass = max_wet_mass / mass_ratio
    print(f"Max dry mass for {TARGET_DELTA_V_KMS} km/s: {max_dry_mass:.0f} tons")

    # Armor budget
    armor_budget = max_dry_mass - base_dry
    print(f"\n{'='*60}")
    print(f"ARMOR BUDGET: {armor_budget:.0f} tons")
    print(f"{'='*60}")

    # Propellant
    propellant = max_wet_mass - max_dry_mass
    print(f"Propellant mass: {propellant:.0f} tons")

    return armor_budget, max_dry_mass, max_wet_mass


def show_armor_options(armor_budget, armors):
    """Show armor distribution options with density-based mass calculation."""

    # Corvette surface area estimates (65m length, 15m width)
    # Nose: conical front ~80 m²
    # Lateral: cylinder surface ~2000 m² (but only partial coverage)
    # Tail: rear ~80 m²
    # These are rough estimates - actual TI values may differ
    ARMOR_AREAS_M2 = {
        'nose': 80,
        'lateral': 400,   # Partial coverage of sides
        'tail': 60,
    }

    print(f"\n{'='*60}")
    print("ARMOR TYPES (Human)")
    print(f"{'='*60}")
    print(f"{'Name':<22} {'Density':>8} {'Baryonic Half':>14} {'1cm mass':>10}")
    print(f"{'':22} {'kg/m³':>8} {'(cm)':>14} {'(tons)':>10}")
    print("-" * 60)

    # Calculate mass per 1cm of armor across all faces
    total_area = sum(ARMOR_AREAS_M2.values())  # m²

    for a in sorted(armors, key=lambda x: x['density']):
        density = a['density']
        baryonic = a['baryonic_half_cm']
        # Mass for 1cm thick armor over all faces
        # Volume = area (m²) × 0.01 (m) = area × 0.01 m³
        # Mass = density (kg/m³) × volume (m³) / 1000 = tons
        mass_1cm = density * total_area * 0.01 / 1000
        print(f"{a['name']:<22} {density:>8} {baryonic:>14.1f} {mass_1cm:>10.1f}")

    print(f"\nTotal armor surface area: {total_area} m² (nose/lateral/tail)")

    print(f"\n{'='*60}")
    print(f"ARMOR DISTRIBUTION (with {armor_budget:.0f}t budget)")
    print(f"{'='*60}")

    # Best armors for different purposes
    # Low density = more thickness for same mass (good for low-density threats)
    # High baryonic half = better kinetic protection per cm

    # Calculate what we can afford with each armor type
    for a in armors:
        density = a['density']
        baryonic = a['baryonic_half_cm']
        name = a['name']

        # Mass per cm of thickness over all faces
        mass_per_cm = density * total_area * 0.01 / 1000  # tons

        if mass_per_cm <= 0:
            continue

        max_cm = armor_budget / mass_per_cm

        # Protection metric: how many "half-values" can we afford?
        half_values = max_cm / baryonic if baryonic > 0 else 0
        damage_reduction = 1 - (0.5 ** half_values)

        print(f"\n{name}:")
        print(f"  Mass per cm: {mass_per_cm:.1f} t")
        print(f"  Max thickness: {max_cm:.1f} cm")
        print(f"  Baryonic half-value: {baryonic:.1f} cm")
        print(f"  Half-values achieved: {half_values:.2f}")
        print(f"  Kinetic damage reduction: {damage_reduction*100:.0f}%")

        # Show distribution (60% front, 30% lateral, 10% tail)
        if max_cm > 0:
            nose_cm = max_cm * 0.6
            lateral_cm = max_cm * 0.3
            tail_cm = max_cm * 0.1
            print(f"  Suggested distribution (60/30/10):")
            print(f"    Nose: {nose_cm:.1f} cm ({nose_cm/baryonic:.1f} half-values)")
            print(f"    Lateral: {lateral_cm:.1f} cm")
            print(f"    Tail: {tail_cm:.1f} cm")


def analyze_combat_survivability(armor_budget, armors):
    """Analyze how armor affects combat survivability."""

    print(f"\n{'='*60}")
    print("COMBAT SURVIVABILITY ANALYSIS")
    print(f"{'='*60}")

    print("\nTypical threats (kinetic):")
    print("  - Coilgun Battery Mk3: ~40kg @ ~10 km/s = ~2 GJ")
    print("  - Heavy Coilgun: ~100kg @ ~10 km/s = ~5 GJ")
    print("  - Spinal Coiler: ~200kg @ ~15 km/s = ~22.5 GJ")

    print("\nArmor effectiveness ranking (for kinetic threats):")
    # Rank by protection per mass: baryonic_half / density
    # Higher = better
    ranked = sorted(armors, key=lambda x: x['baryonic_half_cm'] / x['density'] if x['density'] > 0 else 0, reverse=True)

    print(f"{'Armor':<22} {'Baryonic/Density':>18} {'Half-values/100t':>18}")
    print("-" * 60)
    for a in ranked[:5]:  # Top 5
        if a['density'] > 0:
            efficiency = a['baryonic_half_cm'] / a['density'] * 1000
            # How many half-values per 100t of armor (assuming 540 m² area)
            area = 540
            mass_per_cm = a['density'] * area * 0.01 / 1000
            cm_per_100t = 100 / mass_per_cm if mass_per_cm > 0 else 0
            half_per_100t = cm_per_100t / a['baryonic_half_cm'] if a['baryonic_half_cm'] > 0 else 0
            print(f"{a['name']:<22} {efficiency:>18.2f} {half_per_100t:>18.2f}")


def main():
    print("=" * 60)
    print("ARMOR BUDGET CALCULATOR")
    print("Protium Converter x6 + ICF VII (Exotic)")
    print("=" * 60)

    print("\nShip Configuration:")
    for component, mass in BASE_SHIP_MASS.items():
        print(f"  {component:<15}: {mass:>6} t")

    # Extract armor data
    try:
        armors = extract_armor_data()
        print(f"\nExtracted {len(armors)} human armor types from game files")
    except FileNotFoundError:
        print("\nCouldn't find TI game files, using estimated armor data")
        armors = [
            {'name': 'Steel Armor', 'density': 7850, 'baryonic_half_cm': 7.5, 'xray_half_cm': 2, 'heat_vaporization': 6.8},
            {'name': 'Titanium Armor', 'density': 4820, 'baryonic_half_cm': 10.5, 'xray_half_cm': 3, 'heat_vaporization': 8.9},
            {'name': 'Composite Armor', 'density': 1930, 'baryonic_half_cm': 72, 'xray_half_cm': 5, 'heat_vaporization': 12},
        ]

    # Calculate budget
    armor_budget, max_dry, max_wet = calculate_armor_budget()

    # Show options
    show_armor_options(armor_budget, armors)

    # Combat analysis
    analyze_combat_survivability(armor_budget, armors)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"With {armor_budget:.0f} tons of armor, this corvette can:")
    print(f"  - Maintain 3g combat acceleration")
    print(f"  - Have 1000 km/s delta-v for interplanetary ops")
    print(f"  - Carry significant protective armor")


if __name__ == "__main__":
    main()
