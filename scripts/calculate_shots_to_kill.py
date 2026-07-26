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


# Impact area used by the simulation engine when a torpedo's kinetic penetrator
# strikes armor (see SimulationEngine._resolve_torpedo_hit in src/simulation.py).
TORPEDO_KINETIC_IMPACT_AREA_M2 = 0.1
# flat_chipping the engine uses for that same penetrator (penetrators are focused).
TORPEDO_KINETIC_FLAT_CHIPPING = 0.3


def impact_area_for_projectile_mass(mass_kg: float) -> float:
    """
    Impact/crater area in m^2 for a kinetic projectile of the given mass.

    This MUST mirror SimulationEngine._resolve_projectile_hit in src/simulation.py.
    Ablation is inversely proportional to impact area, so using the old hard-coded
    0.01 m^2 ("10cm x 10cm slug") here while the engine spreads the same energy over
    0.1-0.3 m^2 made this script report armor as ~10-30x less durable than it
    actually is in battle - which is how the published shots-to-kill tables ended up
    far too optimistic.
    """
    area = 0.1 + (mass_kg / 100) * 0.1  # 0.1-0.2 m^2 over the usual mass range
    return min(0.3, max(0.1, area))  # Clamp to 0.1-0.3 m^2


def create_torpedo_weapon(
    fleet_data: dict,
    impact_velocity_kps: float = 5.0,
    weapon_key: str = "torpedo_launcher",
) -> Weapon:
    """Create a torpedo weapon with calculated kinetic energy at given impact velocity.

    Penetrator mass / cooldown / range / magazine are read from fleet_ships.json
    rather than hard-coded, so the table tracks the data the simulation uses.
    """
    torp_data = fleet_data["weapon_types"][weapon_key]
    penetrator_mass_kg = torp_data["penetrator_mass_kg"]
    # KE = 0.5 * m * v^2
    velocity_ms = impact_velocity_kps * 1000
    kinetic_energy_j = 0.5 * penetrator_mass_kg * velocity_ms ** 2
    kinetic_energy_gj = kinetic_energy_j / 1e9

    return Weapon(
        name=f"Torpedo @ {impact_velocity_kps} km/s",
        weapon_type="torpedo",
        kinetic_energy_gj=kinetic_energy_gj,
        cooldown_s=torp_data["cooldown_s"],
        range_km=torp_data["range_km"],
        # Match the engine's torpedo penetrator chipping, not the coilgun value.
        flat_chipping=TORPEDO_KINETIC_FLAT_CHIPPING,
        mass_tons=torp_data["mass_tons"],
        magazine=torp_data["magazine"],
        muzzle_velocity_kps=impact_velocity_kps,
        warhead_mass_kg=penetrator_mass_kg,
        mount=torp_data.get("mount", "hull_turret"),
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

    # Use the same impact area the simulation engine uses for this projectile.
    if weapon.is_missile:
        impact_area_m2 = TORPEDO_KINETIC_IMPACT_AREA_M2
    else:
        impact_area_m2 = impact_area_for_projectile_mass(weapon.warhead_mass_kg)

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
            impact_area_m2=impact_area_m2
        )

        # Check if armor is penetrated
        penetrated = armor_section.is_penetrated()

        # Track first penetration
        if penetrated and not first_penetration:
            shots_to_penetrate = shots_fired
            first_penetration = True

        # If penetrated, apply damage to internal modules
        if penetrated:
            # Energy that actually reaches the hull is what apply_energy_damage
            # returns, exactly as the engine does it (src/simulation.py:
            # `remaining_energy_gj = energy_to_hull_gj`). The old `0.9 * KE`
            # ignored the armor's protection factor and over-damaged internals.
            remaining_damage_gj = energy_to_hull_gj

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


def format_shots(result: SimulationResult) -> str:
    """Render a shots-to-kill cell.

    A result that hit the shot cap was never actually a kill, so print it as
    ">N" instead of an exact-looking count.
    """
    if result.kill_reason == "survived":
        return f">{result.shots_to_kill}"
    return str(result.shots_to_kill)


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

    # Add torpedoes at representative closing speeds.
    #
    # A single "@ 5 km/s" row badly understated them: the Trident carries 14 km/s
    # of its own delta-v, so 5 km/s is slower than it arrives against a STATIONARY
    # target, and impact energy goes as v^2. In a head-on pass the closing speeds
    # of both ships add on top, which is where torpedoes earn their place.
    weapons["torpedo_14kps"] = create_torpedo_weapon(fleet_data, 14.0)   # vs stationary
    weapons["torpedo_26kps"] = create_torpedo_weapon(fleet_data, 26.0)   # head-on pass

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
            print(f"{weapon_display:<25} {format_shots(nose):>8} {format_shots(lat):>8} {format_shots(tail):>8}")

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
            print(f"| {weapon_display} | {format_shots(nose)} | {format_shots(lat)} | {format_shots(tail)} |")

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

        print(f"| {ship_type.capitalize()} | {nose_cm:.0f}/{lat_cm:.0f}/{tail_cm:.0f} | {format_shots(nose)} | {format_shots(lat)} | {format_shots(tail)} |")


if __name__ == "__main__":
    main()
