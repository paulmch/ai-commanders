#!/usr/bin/env python3
"""
Calculate shots-to-kill for all weapon/ship/facing combinations via simulation.

This script simulates combat to determine how many hits are needed to destroy
each ship class from full armor, for each weapon type and hit location.
"""

import sys
from pathlib import Path

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import copy
from dataclasses import dataclass
from src.combat import (
    Weapon, ShipArmor, HitLocation, CombatResolver,
    load_fleet_data, create_weapon_from_fleet_data, create_ship_armor_from_fleet_data
)
from src.modules import ModuleLayout


@dataclass
class SimulationResult:
    """Result of a shots-to-kill simulation."""
    ship_type: str
    weapon_type: str
    location: str
    shots_to_penetrate: int
    shots_to_kill: int
    kill_reason: str  # "reactor", "bridge", or "hull"


def create_torpedo_weapon(impact_velocity_kps: float = 5.0) -> Weapon:
    """Create a torpedo weapon with calculated kinetic energy at given impact velocity."""
    penetrator_mass_kg = 250  # From fleet_ships.json
    # KE = 0.5 * m * v^2
    velocity_ms = impact_velocity_kps * 1000
    kinetic_energy_j = 0.5 * penetrator_mass_kg * velocity_ms ** 2
    kinetic_energy_gj = kinetic_energy_j / 1e9

    return Weapon(
        name=f"Torpedo @ {impact_velocity_kps} km/s",
        weapon_type="torpedo",
        kinetic_energy_gj=kinetic_energy_gj,
        cooldown_s=12.0,
        range_km=2500,
        flat_chipping=0.35,  # Assume similar to coilguns
        mass_tons=40,
        magazine=8,
        muzzle_velocity_kps=impact_velocity_kps,
        warhead_mass_kg=penetrator_mass_kg,
        mount="hull_turret",
        is_missile=True,
    )


def simulate_shots_to_kill(
    weapon: Weapon,
    ship_type: str,
    location: HitLocation,
    fleet_data: dict,
    max_shots: int = 500
) -> SimulationResult:
    """
    Simulate combat until ship is destroyed using energy-based damage.

    Returns the number of shots needed to:
    1. Penetrate armor (first penetrating hit)
    2. Kill the ship (reactor/bridge destroyed or hull critical)
    """
    from src.combat import Armor, HitResult

    # Create fresh armor and module layout
    ship_armor = create_ship_armor_from_fleet_data(fleet_data, ship_type)
    module_layout = ModuleLayout.from_ship_type(ship_type, fleet_data)

    # Get ship structural integrity
    ship_data = fleet_data["ships"][ship_type]
    structural_integrity = ship_data["hull"]["structural_integrity"]

    shots_fired = 0
    shots_to_penetrate = 0
    first_penetration = False
    kill_reason = ""

    armor_section = ship_armor.get_section(location)
    if armor_section is None:
        # No armor - immediate penetration
        return SimulationResult(
            ship_type=ship_type,
            weapon_type=weapon.weapon_type,
            location=location.value,
            shots_to_penetrate=1,
            shots_to_kill=1,
            kill_reason="no_armor"
        )

    while shots_fired < max_shots:
        shots_fired += 1

        # Apply energy-based damage to armor
        ablation_cm, energy_to_hull_gj, chipping = armor_section.apply_energy_damage(
            energy_gj=weapon.kinetic_energy_gj,
            flat_chipping=weapon.flat_chipping,
            impact_area_m2=0.01  # Standard 10cm x 10cm impact
        )

        # Check if armor is penetrated
        penetrated = armor_section.is_penetrated()

        # Track first penetration
        if penetrated and not first_penetration:
            shots_to_penetrate = shots_fired
            first_penetration = True

        # If penetrated, apply damage to internal modules
        if penetrated:
            # Calculate remaining damage (90% of weapon energy after armor breach)
            remaining_damage_gj = weapon.kinetic_energy_gj * 0.9

            # Create a hit result for module damage
            hit_result = HitResult(
                hit=True,
                location=location,
                penetrated=True,
                remaining_damage_gj=remaining_damage_gj
            )

            damage_results = module_layout.apply_penetrating_damage(
                hit_result,
                spread_angle_deg=15.0
            )

        # Check for kill conditions
        # 1. Critical module destroyed (reactor or bridge)
        if module_layout.has_critical_damage:
            for module in module_layout.get_critical_modules():
                if module.is_destroyed:
                    kill_reason = module.module_type.value
                    break
            break

        # 2. Hull integrity critical (below 25%)
        if module_layout.ship_integrity_percent < 25.0:
            kill_reason = "hull"
            break

    # If we never penetrated, set penetration to max
    if not first_penetration:
        shots_to_penetrate = shots_fired

    return SimulationResult(
        ship_type=ship_type,
        weapon_type=weapon.weapon_type,
        location=location.value,
        shots_to_penetrate=shots_to_penetrate,
        shots_to_kill=shots_fired if kill_reason else max_shots,
        kill_reason=kill_reason or "survived"
    )


def main():
    # Load fleet data
    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    fleet_data = load_fleet_data(data_path)

    # Ship types to test
    ship_types = ["corvette", "frigate", "destroyer", "cruiser", "battlecruiser", "battleship", "dreadnought"]

    # Weapons to test
    weapon_types = [
        "spinal_coiler_mk3",
        "heavy_siege_coiler_mk3",
        "heavy_coilgun_mk3",
        "coilgun_mk3",
        "light_coilgun_mk3",
    ]

    # Create weapons
    weapons = {}
    for wtype in weapon_types:
        weapons[wtype] = create_weapon_from_fleet_data(fleet_data, wtype)

    # Add torpedo
    weapons["torpedo_5kps"] = create_torpedo_weapon(5.0)

    # Hit locations
    locations = [HitLocation.NOSE, HitLocation.LATERAL, HitLocation.TAIL]

    # Run simulations - store results by (ship, weapon_key, location)
    results = {}

    print("Running shots-to-kill simulations...")
    print("=" * 80)

    for ship_type in ship_types:
        print(f"\n{ship_type.upper()}")
        print("-" * 40)

        for weapon_key, weapon in weapons.items():
            for location in locations:
                result = simulate_shots_to_kill(
                    weapon=weapon,
                    ship_type=ship_type,
                    location=location,
                    fleet_data=fleet_data
                )
                results[(ship_type, weapon_key, location.value)] = result

        # Print table for this ship
        print(f"\n{'Weapon':<25} {'Nose':>8} {'Lateral':>8} {'Tail':>8}")
        print("-" * 55)

        for weapon_key in list(weapons.keys()):
            nose = results[(ship_type, weapon_key, "nose")]
            lat = results[(ship_type, weapon_key, "lateral")]
            tail = results[(ship_type, weapon_key, "tail")]

            weapon_display = weapons[weapon_key].name[:24]
            print(f"{weapon_display:<25} {nose.shots_to_kill:>8} {lat.shots_to_kill:>8} {tail.shots_to_kill:>8}")

    # Generate markdown tables
    print("\n\n" + "=" * 80)
    print("MARKDOWN OUTPUT FOR docs/ships.md")
    print("=" * 80)

    # Shots to Kill (combined penetration + destruction)
    print("\n### Shots to Kill (from full armor)\n")
    print("Number of hits required to destroy each ship class, starting from full armor.\n")

    for ship_type in ship_types:
        # Get armor thicknesses for header
        ship_data = fleet_data["ships"][ship_type]
        armor_data = ship_data["armor"]["sections"]
        nose_cm = armor_data["nose"]["thickness_cm"]
        lat_cm = armor_data["lateral"]["thickness_cm"]
        tail_cm = armor_data["tail"]["thickness_cm"]

        print(f"\n#### vs {ship_type.capitalize()} (Armor: {nose_cm:.0f}/{lat_cm:.0f}/{tail_cm:.0f} cm)\n")
        print("| Weapon | Nose | Lateral | Tail |")
        print("|--------|------|---------|------|")

        for weapon_key in list(weapons.keys()):
            nose = results[(ship_type, weapon_key, "nose")]
            lat = results[(ship_type, weapon_key, "lateral")]
            tail = results[(ship_type, weapon_key, "tail")]

            weapon_display = weapons[weapon_key].name[:25]
            print(f"| {weapon_display} | {nose.shots_to_kill} | {lat.shots_to_kill} | {tail.shots_to_kill} |")

    # Summary table - all ships, spinal coiler only
    print("\n### Summary: Shots to Kill by Ship Class\n")
    print("Using Spinal Coiler Mk3 (4.29 GJ per shot):\n")
    print("| Ship | Armor (N/L/T cm) | Nose | Lateral | Tail |")
    print("|------|-----------------|------|---------|------|")

    for ship_type in ship_types:
        ship_data = fleet_data["ships"][ship_type]
        armor_data = ship_data["armor"]["sections"]
        nose_cm = armor_data["nose"]["thickness_cm"]
        lat_cm = armor_data["lateral"]["thickness_cm"]
        tail_cm = armor_data["tail"]["thickness_cm"]

        nose = results[(ship_type, "spinal_coiler_mk3", "nose")]
        lat = results[(ship_type, "spinal_coiler_mk3", "lateral")]
        tail = results[(ship_type, "spinal_coiler_mk3", "tail")]

        print(f"| {ship_type.capitalize()} | {nose_cm:.0f}/{lat_cm:.0f}/{tail_cm:.0f} | {nose.shots_to_kill} | {lat.shots_to_kill} | {tail.shots_to_kill} |")


if __name__ == "__main__":
    main()
