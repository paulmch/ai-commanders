#!/usr/bin/env python3
"""
Extract ship hull and module data from Terra Invicta game files.
Outputs markdown tables and raw JSON.
"""

import json
from pathlib import Path

# Game file paths
GAME_BASE = Path("/mnt/d/SteamLibrary/steamapps/common/Terra Invicta/TerraInvicta_Data/StreamingAssets/Templates")
HULL_FILE = GAME_BASE / "TIShipHullTemplate.json"
UTILITY_FILE = GAME_BASE / "TIUtilityModuleTemplate.json"
HEATSINK_FILE = GAME_BASE / "TIHeatSinkTemplate.json"
RADIATOR_FILE = GAME_BASE / "TIRadiatorTemplate.json"
BATTERY_FILE = GAME_BASE / "TIBatteryTemplate.json"

# Output paths
OUTPUT_DIR = Path("/home/plmch/ai-commanders/data")
OUTPUT_MD = OUTPUT_DIR / "modules.md"
OUTPUT_JSON = OUTPUT_DIR / "modules.json"


def load_json(path: Path) -> list:
    """Load JSON file and return list of objects."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_hulls(data: list) -> list[dict]:
    """Extract hull data: name, mass_tons, structuralIntegrity, crew, hardpoints."""
    hulls = []
    for item in data:
        hull = {
            "name": item.get("friendlyName", item.get("dataName", "Unknown")),
            "dataName": item.get("dataName", ""),
            "mass_tons": item.get("mass_tons", 0),
            "structuralIntegrity": item.get("structuralIntegrity", 0),
            "crew": item.get("crew", 0),
            "noseHardpoints": item.get("noseHardpoints", 0),
            "hullHardpoints": item.get("hullHardpoints", 0),
            "internalModules": item.get("internalModules", 0),
            "consTier": item.get("consTier", 0),
            "alien": item.get("alien", False),
            "length_m": item.get("length_m", 0),
        }
        hulls.append(hull)
    return hulls


def extract_utility_modules(data: list) -> list[dict]:
    """Extract utility module data: name, mass_tons, power, special effects."""
    modules = []
    for item in data:
        # Skip the "Empty" module
        if item.get("dataName") == "Empty":
            continue
        module = {
            "name": item.get("friendlyName", item.get("dataName", "Unknown")),
            "dataName": item.get("dataName", ""),
            "mass_tons": item.get("mass_tons", 0),
            "crew": item.get("crew", 0),
            "powerRequirement_MW": item.get("powerRequirement_MW", 0),
            "specialModuleRules": item.get("specialModuleRules", []),
            "specialModuleValue": item.get("specialModuleValue", None),
            "grouping": item.get("grouping", -1),
        }
        modules.append(module)
    return modules


def extract_heatsinks(data: list) -> list[dict]:
    """Extract heatsink data: name, capacity_GJ, mass_tons, specific capacity."""
    heatsinks = []
    for item in data:
        capacity = item.get("heatCapacity_GJ", 0)
        mass = item.get("mass_tons", 1)
        specific_capacity = round(capacity / mass, 2) if mass > 0 else 0
        heatsink = {
            "name": item.get("displayName", item.get("dataName", "Unknown")),
            "dataName": item.get("dataName", ""),
            "capacity_GJ": capacity,
            "mass_tons": mass,
            "specific_capacity_GJ_ton": specific_capacity,
            "crew": item.get("crew", 0),
        }
        heatsinks.append(heatsink)
    return heatsinks


def extract_radiators(data: list) -> list[dict]:
    """Extract radiator data: name, specificPower_kW_kg, operatingTemp, vulnerability."""
    radiators = []
    for item in data:
        radiator = {
            "name": item.get("friendlyName", item.get("dataName", "Unknown")),
            "dataName": item.get("dataName", ""),
            "specificPower_kW_kg": item.get("specificPower_2s_KWkg", 0),
            "operatingTemp_K": item.get("operatingTemp_K", 0),
            "vulnerability": item.get("vulnerability", 0),
            "emissivity": item.get("emissivity", 0),
            "collector": item.get("collector", False),
            "radiatorType": item.get("radiatorType", ""),
            "crew": item.get("crew", 0),
        }
        radiators.append(radiator)
    return radiators


def extract_batteries(data: list) -> list[dict]:
    """Extract battery data: name, capacity_GJ, mass_tons, rechargeRate."""
    batteries = []
    for item in data:
        # Skip disabled batteries
        if item.get("disable", False):
            continue
        capacity = item.get("energyCapacity_GJ", 0)
        mass = item.get("mass_tons", 1)
        specific_capacity = round(capacity / mass, 2) if mass > 0 else 0
        battery = {
            "name": item.get("friendlyName", item.get("dataName", "Unknown")),
            "dataName": item.get("dataName", ""),
            "capacity_GJ": capacity,
            "mass_tons": mass,
            "specific_capacity_GJ_ton": specific_capacity,
            "rechargeRate_GJ_s": item.get("rechargeRate_GJs", 0),
            "crew": item.get("crew", 0),
            "hp": item.get("hp", 0),
        }
        batteries.append(battery)
    return batteries


def generate_markdown(hulls: list, utilities: list, heatsinks: list, radiators: list, batteries: list) -> str:
    """Generate markdown document with tables for all module types."""
    lines = ["# Terra Invicta Ship Modules Data\n"]

    # Hulls table
    lines.append("## Ship Hulls\n")
    lines.append("| Name | Mass (tons) | Structural Integrity | Crew | Nose HP | Hull HP | Utility Slots | Tier | Length (m) | Alien |")
    lines.append("|------|-------------|---------------------|------|---------|---------|---------------|------|------------|-------|")
    for h in sorted(hulls, key=lambda x: (x["alien"], x["consTier"], x["mass_tons"])):
        lines.append(f"| {h['name']} | {h['mass_tons']} | {h['structuralIntegrity']} | {h['crew']} | {h['noseHardpoints']} | {h['hullHardpoints']} | {h['internalModules']} | {h['consTier']} | {h['length_m']} | {'Yes' if h['alien'] else 'No'} |")
    lines.append("")

    # Utility modules table
    lines.append("## Utility Modules\n")
    lines.append("| Name | Mass (tons) | Crew | Power (MW) | Special Rules | Value |")
    lines.append("|------|-------------|------|------------|---------------|-------|")
    for m in sorted(utilities, key=lambda x: (x["grouping"], x["name"])):
        rules = ", ".join(m["specialModuleRules"]) if m["specialModuleRules"] else "-"
        value = m["specialModuleValue"] if m["specialModuleValue"] is not None else "-"
        lines.append(f"| {m['name']} | {m['mass_tons']} | {m['crew']} | {m['powerRequirement_MW']} | {rules} | {value} |")
    lines.append("")

    # Heatsinks table
    lines.append("## Heat Sinks\n")
    lines.append("| Name | Capacity (GJ) | Mass (tons) | Specific Capacity (GJ/ton) | Crew |")
    lines.append("|------|---------------|-------------|---------------------------|------|")
    for h in sorted(heatsinks, key=lambda x: x["specific_capacity_GJ_ton"]):
        lines.append(f"| {h['name']} | {h['capacity_GJ']} | {h['mass_tons']} | {h['specific_capacity_GJ_ton']} | {h['crew']} |")
    lines.append("")

    # Radiators table
    lines.append("## Radiators\n")
    lines.append("| Name | Specific Power (kW/kg) | Operating Temp (K) | Vulnerability | Emissivity | Type | Collector | Crew |")
    lines.append("|------|------------------------|-------------------|---------------|------------|------|-----------|------|")
    for r in sorted(radiators, key=lambda x: x["specificPower_kW_kg"]):
        collector = "Yes" if r["collector"] else "No"
        lines.append(f"| {r['name']} | {r['specificPower_kW_kg']} | {r['operatingTemp_K']} | {r['vulnerability']} | {r['emissivity']} | {r['radiatorType']} | {collector} | {r['crew']} |")
    lines.append("")

    # Batteries table
    lines.append("## Batteries\n")
    lines.append("| Name | Capacity (GJ) | Mass (tons) | Specific Capacity (GJ/ton) | Recharge Rate (GJ/s) | Crew | HP |")
    lines.append("|------|---------------|-------------|---------------------------|---------------------|------|-----|")
    for b in sorted(batteries, key=lambda x: x["specific_capacity_GJ_ton"]):
        lines.append(f"| {b['name']} | {b['capacity_GJ']} | {b['mass_tons']} | {b['specific_capacity_GJ_ton']} | {b['rechargeRate_GJ_s']} | {b['crew']} | {b['hp']} |")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    print("Loading game data files...")

    # Load all data
    hulls_raw = load_json(HULL_FILE)
    utilities_raw = load_json(UTILITY_FILE)
    heatsinks_raw = load_json(HEATSINK_FILE)
    radiators_raw = load_json(RADIATOR_FILE)
    batteries_raw = load_json(BATTERY_FILE)

    print(f"  Hulls: {len(hulls_raw)} entries")
    print(f"  Utility modules: {len(utilities_raw)} entries")
    print(f"  Heat sinks: {len(heatsinks_raw)} entries")
    print(f"  Radiators: {len(radiators_raw)} entries")
    print(f"  Batteries: {len(batteries_raw)} entries")

    # Extract relevant data
    print("\nExtracting module data...")
    hulls = extract_hulls(hulls_raw)
    utilities = extract_utility_modules(utilities_raw)
    heatsinks = extract_heatsinks(heatsinks_raw)
    radiators = extract_radiators(radiators_raw)
    batteries = extract_batteries(batteries_raw)

    # Generate combined data structure
    combined_data = {
        "hulls": hulls,
        "utility_modules": utilities,
        "heatsinks": heatsinks,
        "radiators": radiators,
        "batteries": batteries,
    }

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write JSON output
    print(f"\nWriting JSON to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    # Generate and write markdown
    print(f"Writing markdown to {OUTPUT_MD}...")
    markdown = generate_markdown(hulls, utilities, heatsinks, radiators, batteries)
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print("\nExtraction complete!")
    print(f"  Hulls extracted: {len(hulls)}")
    print(f"  Utility modules extracted: {len(utilities)}")
    print(f"  Heat sinks extracted: {len(heatsinks)}")
    print(f"  Radiators extracted: {len(radiators)}")
    print(f"  Batteries extracted: {len(batteries)}")


if __name__ == "__main__":
    main()
