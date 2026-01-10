#!/usr/bin/env python3
"""
Extract drive data from Terra Invicta game files.
Filters out alien drives (those requiring exotics > 0).
"""

import json
from pathlib import Path

# Paths
GAME_PATH = Path("/mnt/d/SteamLibrary/steamapps/common/Terra Invicta/TerraInvicta_Data/StreamingAssets/Templates/TIDriveTemplate.json")
OUTPUT_DIR = Path("/home/plmch/ai-commanders/data")
OUTPUT_MD = OUTPUT_DIR / "drives.md"
OUTPUT_JSON = OUTPUT_DIR / "drives.json"


def is_human_drive(drive: dict) -> bool:
    """Check if drive is human (no exotics required)."""
    build_materials = drive.get("weightedBuildMaterials", {})
    exotics = build_materials.get("exotics", 0)
    return exotics == 0


def parse_float(value) -> float:
    """Parse a float that may have comma as thousand separator."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.replace(",", ""))
    return 0.0


def extract_drive_data(drive: dict) -> dict:
    """Extract relevant fields from a drive."""
    specific_power = parse_float(drive.get("specificPower_kgMW", 0))
    thrust_rating_gw = parse_float(drive.get("thrustRating_GW", 0))

    # Calculate actual drive mass
    if specific_power > 0:
        # specificPower_kgMW is kg per MW, thrustRating_GW is in GW
        # Convert GW to MW (multiply by 1000) then multiply by kg/MW
        actual_mass_tons = (specific_power * thrust_rating_gw * 1000) / 1000  # kg to tons
    else:
        actual_mass_tons = 0

    return {
        "name": drive.get("friendlyName", drive.get("dataName", "Unknown")),
        "dataName": drive.get("dataName", ""),
        "thrust_N": drive.get("thrust_N", 0),
        "EV_kps": drive.get("EV_kps", 0),
        "flatMass_tons": drive.get("flatMass_tons", 0),
        "specificPower_kgMW": specific_power,
        "efficiency": drive.get("efficiency", 0),
        "requiredPowerPlant": drive.get("requiredPowerPlant", ""),
        "thrustRating_GW": thrust_rating_gw,
        "cooling": drive.get("cooling", ""),
        "powerGen": drive.get("powerGen", ""),
        "propellant": drive.get("propellant", ""),
        "actualMass_tons": actual_mass_tons,
        "driveClassification": drive.get("driveClassification", ""),
        "thrusters": drive.get("thrusters", 1),
    }


def format_thrust(thrust_n: int) -> str:
    """Format thrust in human readable format."""
    if thrust_n >= 1_000_000:
        return f"{thrust_n / 1_000_000:.2f} MN"
    elif thrust_n >= 1000:
        return f"{thrust_n / 1000:.1f} kN"
    return f"{thrust_n} N"


def main():
    # Read game data
    print(f"Reading: {GAME_PATH}")
    with open(GAME_PATH, "r", encoding="utf-8") as f:
        drives = json.load(f)

    print(f"Total drives in file: {len(drives)}")

    # Filter human drives and extract data
    human_drives = []
    for drive in drives:
        if is_human_drive(drive):
            human_drives.append(extract_drive_data(drive))

    print(f"Human drives (no exotics): {len(human_drives)}")

    # Sort by thrust descending
    human_drives.sort(key=lambda x: x["thrust_N"], reverse=True)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON
    print(f"Writing: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(human_drives, f, indent=2)

    # Generate markdown table
    print(f"Writing: {OUTPUT_MD}")
    md_lines = [
        "# Terra Invicta Human Drives",
        "",
        "Extracted from TIDriveTemplate.json. Sorted by thrust (descending).",
        "Excludes alien drives (those requiring exotics > 0).",
        "",
        "| Name | Thrust | EV (km/s) | Flat Mass (t) | Specific Power (kg/MW) | Actual Mass (t) | Efficiency | Power Plant | Thrust Rating (GW) | Cooling | Power Gen | Propellant | Classification |",
        "|------|--------|-----------|---------------|------------------------|-----------------|------------|-------------|-------------------|---------|-----------|------------|----------------|",
    ]

    for d in human_drives:
        thrust_str = format_thrust(d["thrust_N"])
        actual_mass = f"{d['actualMass_tons']:.1f}" if d['actualMass_tons'] > 0 else "-"
        row = (
            f"| {d['name']} "
            f"| {thrust_str} "
            f"| {d['EV_kps']} "
            f"| {d['flatMass_tons']} "
            f"| {d['specificPower_kgMW']} "
            f"| {actual_mass} "
            f"| {d['efficiency']} "
            f"| {d['requiredPowerPlant']} "
            f"| {d['thrustRating_GW']:.3f} "
            f"| {d['cooling']} "
            f"| {d['powerGen']} "
            f"| {d['propellant']} "
            f"| {d['driveClassification']} |"
        )
        md_lines.append(row)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nDone! Extracted {len(human_drives)} human drives.")
    print(f"  Markdown: {OUTPUT_MD}")
    print(f"  JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
