#!/usr/bin/env python3
"""
Extract reactor/power plant data from Terra Invicta game files.
"""

import json
from pathlib import Path


def extract_reactors():
    # Paths
    game_path = Path("/mnt/d/SteamLibrary/steamapps/common/Terra Invicta/TerraInvicta_Data/StreamingAssets/Templates/TIPowerPlantTemplate.json")
    output_dir = Path("/home/plmch/ai-commanders/data")

    # Read the game data
    with open(game_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract relevant fields for each reactor
    reactors = []
    for reactor in raw_data:
        name = reactor.get("friendlyName", reactor.get("dataName", "Unknown"))
        max_output = reactor.get("maxOutput_GW", 0)
        specific_power = reactor.get("specificPower_tGW", 0)
        efficiency = reactor.get("efficiency", 0)
        power_plant_class = reactor.get("powerPlantClass", "Unknown")
        crew = reactor.get("crew", 0)

        # Calculate mass: output * specificPower (tons)
        calculated_mass = max_output * specific_power

        # Check if requires exotics
        build_materials = reactor.get("weightedBuildMaterials", {})
        requires_exotics = build_materials.get("exotics", 0) > 0
        requires_antimatter = build_materials.get("antimatter", 0) > 0

        reactors.append({
            "name": name,
            "dataName": reactor.get("dataName", ""),
            "maxOutput_GW": max_output,
            "specificPower_tGW": specific_power,
            "efficiency": efficiency,
            "powerPlantClass": power_plant_class,
            "crew": crew,
            "calculated_mass_t": calculated_mass,
            "requires_exotics": requires_exotics,
            "requires_antimatter": requires_antimatter,
        })

    # Sort by output descending
    reactors.sort(key=lambda x: x["maxOutput_GW"], reverse=True)

    # Save raw JSON
    json_path = output_dir / "reactors.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reactors, f, indent=2)
    print(f"Saved JSON to: {json_path}")

    # Generate markdown table
    md_lines = [
        "# Terra Invicta Reactors/Power Plants",
        "",
        "Sorted by maximum output (descending).",
        "",
        "| Name | Output (GW) | Specific Power (t/GW) | Efficiency | Class | Crew | Mass (t) | Exotics | Antimatter |",
        "|------|-------------|----------------------|------------|-------|------|----------|---------|------------|",
    ]

    for r in reactors:
        exotics_str = "Yes" if r["requires_exotics"] else "No"
        antimatter_str = "Yes" if r["requires_antimatter"] else "No"
        md_lines.append(
            f"| {r['name']} | {r['maxOutput_GW']:,.2f} | {r['specificPower_tGW']:.6f} | {r['efficiency']:.4f} | {r['powerPlantClass']} | {r['crew']} | {r['calculated_mass_t']:,.2f} | {exotics_str} | {antimatter_str} |"
        )

    md_path = output_dir / "reactors.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved Markdown to: {md_path}")

    print(f"\nTotal reactors extracted: {len(reactors)}")


if __name__ == "__main__":
    extract_reactors()
