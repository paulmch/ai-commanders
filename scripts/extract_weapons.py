#!/usr/bin/env python3
"""
Extract weapon data from Terra Invicta game files and save to readable formats.
"""

import json
from pathlib import Path
from typing import Any

# Game file paths
MAGNETIC_GUNS_PATH = Path("/mnt/d/SteamLibrary/steamapps/common/Terra Invicta/TerraInvicta_Data/StreamingAssets/Templates/TIMagneticGunTemplate.json")
LASERS_PATH = Path("/mnt/d/SteamLibrary/steamapps/common/Terra Invicta/TerraInvicta_Data/StreamingAssets/Templates/TILaserWeaponTemplate.json")
MISSILES_PATH = Path("/mnt/d/SteamLibrary/steamapps/common/Terra Invicta/TerraInvicta_Data/StreamingAssets/Templates/TIMissileTemplate.json")

# Output paths
OUTPUT_DIR = Path("/home/plmch/ai-commanders/data")
WEAPONS_MD_PATH = OUTPUT_DIR / "weapons.md"
WEAPONS_JSON_PATH = OUTPUT_DIR / "weapons.json"


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load JSON file and return list of weapon entries."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_kinetic_energy_mj(mass_kg: float, velocity_kps: float) -> float:
    """
    Calculate kinetic energy in MJ.
    KE = 0.5 * m * v^2
    mass in kg, velocity in km/s (convert to m/s)
    Result in MJ
    """
    velocity_mps = velocity_kps * 1000  # km/s to m/s
    energy_j = 0.5 * mass_kg * velocity_mps ** 2
    energy_mj = energy_j / 1_000_000  # J to MJ
    return round(energy_mj, 2)


def extract_magnetic_guns(data: list[dict]) -> list[dict]:
    """Extract relevant fields from magnetic gun data."""
    weapons = []
    for w in data:
        mass_kg = w.get("ammoMass_kg", 0)
        velocity_kps = w.get("muzzleVelocity_kps", 0)
        kinetic_energy_mj = calculate_kinetic_energy_mj(mass_kg, velocity_kps)

        weapons.append({
            "name": w.get("friendlyName", w.get("dataName", "Unknown")),
            "dataName": w.get("dataName", ""),
            "mass_tons": w.get("baseWeaponMass_tons", 0),
            "magazine": w.get("magazine", 0),
            "ammoMass_kg": mass_kg,
            "muzzleVelocity_kps": velocity_kps,
            "range_km": w.get("targetingRange_km", 0),
            "cooldown_s": w.get("cooldown_s", 0),
            "salvoSize": w.get("salvo_shots", 1),
            "kinetic_energy_MJ": kinetic_energy_mj,
            "attackMode": w.get("attackMode", False),
            "defenseMode": w.get("defenseMode", False),
            "mount": w.get("mount", ""),
        })
    return weapons


def extract_lasers(data: list[dict]) -> list[dict]:
    """Extract relevant fields from laser weapon data."""
    weapons = []
    for w in data:
        weapons.append({
            "name": w.get("friendlyName", w.get("dataName", "Unknown")),
            "dataName": w.get("dataName", ""),
            "mass_tons": w.get("baseWeaponMass_tons", 0),
            "shotPower_MJ": w.get("shotPower_MJ", 0),
            "range_km": w.get("targetingRange_km", 0),
            "cooldown_s": w.get("cooldown_s", 0),
            "efficiency": w.get("efficiency", 0),
            "defenseMode": w.get("defenseMode", False),
            "attackMode": w.get("attackMode", False),
            "mount": w.get("mount", ""),
            "wavelength_nm": w.get("wavelength_nm", 0),
            "mirrorRadius_cm": w.get("mirrorRadius_cm", 0),
        })
    return weapons


def extract_missiles(data: list[dict]) -> list[dict]:
    """Extract relevant fields from missile/torpedo data."""
    weapons = []
    for w in data:
        weapons.append({
            "name": w.get("friendlyName", w.get("dataName", "Unknown")),
            "dataName": w.get("dataName", ""),
            "mass_kg": w.get("ammoMass_kg", 0),
            "acceleration_G": w.get("acceleration_g", 0),
            "deltaV_kps": w.get("deltaV_kps", 0),
            "damage_MJ": w.get("flatDamage_MJ", 0),
            "magazine": w.get("magazine", 0),
            "warheadMass_kg": w.get("warheadMass_kg", 0),
            "warheadClass": w.get("warheadClass", ""),
            "range_km": w.get("targetingRange_km", 0),
            "cooldown_s": w.get("cooldown_s", 0),
            "salvo_shots": w.get("salvo_shots", 1),
            "attackMode": w.get("attackMode", False),
            "defenseMode": w.get("defenseMode", False),
            "mount": w.get("mount", ""),
        })
    return weapons


def generate_markdown(magnetic_guns: list[dict], lasers: list[dict], missiles: list[dict]) -> str:
    """Generate markdown content with weapon tables."""
    lines = []
    lines.append("# Terra Invicta Weapon Data")
    lines.append("")
    lines.append("Extracted from game template files.")
    lines.append("")

    # Magnetic Guns Section
    lines.append("## Magnetic Guns (Railguns/Coilguns)")
    lines.append("")
    lines.append("| Name | Mass (tons) | Magazine | Ammo Mass (kg) | Velocity (km/s) | Range (km) | Cooldown (s) | Kinetic Energy (MJ) | Attack | Defense |")
    lines.append("|------|-------------|----------|----------------|-----------------|------------|--------------|---------------------|--------|---------|")

    for w in magnetic_guns:
        attack = "Yes" if w["attackMode"] else "No"
        defense = "Yes" if w["defenseMode"] else "No"
        lines.append(f"| {w['name']} | {w['mass_tons']} | {w['magazine']} | {w['ammoMass_kg']} | {w['muzzleVelocity_kps']} | {w['range_km']} | {w['cooldown_s']} | {w['kinetic_energy_MJ']} | {attack} | {defense} |")

    lines.append("")
    lines.append(f"**Total: {len(magnetic_guns)} magnetic guns**")
    lines.append("")

    # Lasers Section
    lines.append("## Laser Weapons")
    lines.append("")
    lines.append("| Name | Mass (tons) | Shot Power (MJ) | Range (km) | Cooldown (s) | Efficiency | Attack | Defense |")
    lines.append("|------|-------------|-----------------|------------|--------------|------------|--------|---------|")

    for w in lasers:
        attack = "Yes" if w["attackMode"] else "No"
        defense = "Yes" if w["defenseMode"] else "No"
        lines.append(f"| {w['name']} | {w['mass_tons']} | {w['shotPower_MJ']} | {w['range_km']} | {w['cooldown_s']} | {w['efficiency']} | {attack} | {defense} |")

    lines.append("")
    lines.append(f"**Total: {len(lasers)} laser weapons**")
    lines.append("")

    # Missiles Section
    lines.append("## Missiles and Torpedoes")
    lines.append("")
    lines.append("| Name | Mass (kg) | Acceleration (G) | Delta-V (km/s) | Damage (MJ) | Magazine | Warhead Class | Attack | Defense |")
    lines.append("|------|-----------|------------------|----------------|-------------|----------|---------------|--------|---------|")

    for w in missiles:
        attack = "Yes" if w["attackMode"] else "No"
        defense = "Yes" if w["defenseMode"] else "No"
        lines.append(f"| {w['name']} | {w['mass_kg']} | {w['acceleration_G']} | {w['deltaV_kps']} | {w['damage_MJ']} | {w['magazine']} | {w['warheadClass']} | {attack} | {defense} |")

    lines.append("")
    lines.append(f"**Total: {len(missiles)} missiles/torpedoes**")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main function to extract and save weapon data."""
    print("Loading weapon data from Terra Invicta game files...")

    # Load raw data
    magnetic_guns_raw = load_json(MAGNETIC_GUNS_PATH)
    lasers_raw = load_json(LASERS_PATH)
    missiles_raw = load_json(MISSILES_PATH)

    print(f"  Loaded {len(magnetic_guns_raw)} magnetic guns")
    print(f"  Loaded {len(lasers_raw)} laser weapons")
    print(f"  Loaded {len(missiles_raw)} missiles/torpedoes")

    # Extract relevant fields
    magnetic_guns = extract_magnetic_guns(magnetic_guns_raw)
    lasers = extract_lasers(lasers_raw)
    missiles = extract_missiles(missiles_raw)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate and save markdown
    markdown_content = generate_markdown(magnetic_guns, lasers, missiles)
    with open(WEAPONS_MD_PATH, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Saved markdown to: {WEAPONS_MD_PATH}")

    # Save JSON
    all_weapons = {
        "magnetic_guns": magnetic_guns,
        "lasers": lasers,
        "missiles": missiles,
    }
    with open(WEAPONS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_weapons, f, indent=2)
    print(f"Saved JSON to: {WEAPONS_JSON_PATH}")

    print("Done!")


if __name__ == "__main__":
    main()
