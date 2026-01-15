#!/usr/bin/env python3
"""Test script to compare Admiral and Captain prompts side by side."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.admiral import (
    AdmiralSnapshot,
    FriendlyShipSnapshot,
    EnemyShipSnapshot,
    ProjectileSnapshot,
    AdmiralOrder,
)
from src.llm.prompts import build_captain_prompt, format_admiral_orders_for_captain
from src.llm.battle_runner import load_fleet_data
from src.simulation import CombatSimulation, create_ship_from_fleet_data
from src.physics import Vector3D


def main():
    fleet_data = load_fleet_data()

    # Create a real simulation
    sim = CombatSimulation()

    # Add alpha ships
    alpha_1 = create_ship_from_fleet_data(
        ship_id="alpha_1",
        ship_type="destroyer",
        faction="alpha",
        fleet_data=fleet_data,
        position=Vector3D(-200000, 0, 0),  # -200 km
        velocity=Vector3D(2500, 0, 0),  # 2.5 km/s
        forward=Vector3D(1, 0, 0),
    )
    alpha_1.name = "TIS Haiku-1"
    sim.add_ship(alpha_1)

    alpha_2 = create_ship_from_fleet_data(
        ship_id="alpha_2",
        ship_type="destroyer",
        faction="alpha",
        fleet_data=fleet_data,
        position=Vector3D(-200000, 5000, 0),
        velocity=Vector3D(2300, 0, 0),
        forward=Vector3D(1, 0, 0),
    )
    alpha_2.name = "TIS Haiku-2"
    sim.add_ship(alpha_2)

    alpha_3 = create_ship_from_fleet_data(
        ship_id="alpha_3",
        ship_type="dreadnought",
        faction="alpha",
        fleet_data=fleet_data,
        position=Vector3D(-200000, 10000, 0),
        velocity=Vector3D(1500, 0, 0),
        forward=Vector3D(1, 0, 0),
    )
    alpha_3.name = "TIS Haiku-3"
    sim.add_ship(alpha_3)

    # Add beta ships
    beta_1 = create_ship_from_fleet_data(
        ship_id="beta_1",
        ship_type="destroyer",
        faction="beta",
        fleet_data=fleet_data,
        position=Vector3D(200000, 0, 0),
        velocity=Vector3D(-2400, 0, 0),
        forward=Vector3D(-1, 0, 0),
    )
    beta_1.name = "OCS Gemini-1"
    sim.add_ship(beta_1)

    beta_2 = create_ship_from_fleet_data(
        ship_id="beta_2",
        ship_type="destroyer",
        faction="beta",
        fleet_data=fleet_data,
        position=Vector3D(200000, 5000, 0),
        velocity=Vector3D(-2200, 0, 0),
        forward=Vector3D(-1, 0, 0),
    )
    beta_2.name = "OCS Gemini-2"
    sim.add_ship(beta_2)

    beta_3 = create_ship_from_fleet_data(
        ship_id="beta_3",
        ship_type="dreadnought",
        faction="beta",
        fleet_data=fleet_data,
        position=Vector3D(200000, 10000, 0),
        velocity=Vector3D(-1200, 0, 0),
        forward=Vector3D(-1, 0, 0),
    )
    beta_3.name = "OCS Gemini-3"
    sim.add_ship(beta_3)

    print("=" * 80)
    print("SIMULATION SHIP DATA")
    print("=" * 80)

    for ship_id, ship in sim.ships.items():
        print(f"\nShip ID: {ship_id}")
        print(f"  Name: {ship.name}")
        print(f"  Faction: {ship.faction}")
        print(f"  Type: {ship.ship_type}")

    print("\n" + "=" * 80)
    print("CAPTAIN PROMPT - What does the captain see?")
    print("=" * 80)

    # Build captain prompt for alpha_1
    captain_prompt = build_captain_prompt(
        ship=alpha_1,
        simulation=sim,
        captain_name="Captain Haiku-1",
        ship_name="TIS Haiku-1",
        personality_text="I am a tactical commander.",
        fleet_data=fleet_data,
    )

    print(captain_prompt)

    print("\n" + "=" * 80)
    print("ADMIRAL ORDERS SECTION - What does captain see from Admiral?")
    print("=" * 80)

    # Create mock Admiral orders
    admiral_orders = [
        AdmiralOrder(
            target_ship_id="alpha_1",
            target_ship_name="TIS Haiku-1",
            order_text="Target beta_1. INTERCEPT at full throttle. Fire spinal when aligned.",
            priority="HIGH",
            suggested_target="beta_1",
        ),
    ]

    orders_section = format_admiral_orders_for_captain(
        orders=admiral_orders,
        fleet_directive="Focus fire on enemy destroyers.",
    )

    print(orders_section)

    print("\n" + "=" * 80)
    print("KEY QUESTION: Where do ship IDs appear?")
    print("=" * 80)

    # Search for IDs in the captain prompt
    print("\nSearching captain prompt for 'alpha_' or 'beta_'...")
    for i, line in enumerate(captain_prompt.split('\n')):
        if 'alpha_' in line.lower() or 'beta_' in line.lower():
            print(f"  Line {i}: {line}")

    print("\nSearching for ship names...")
    for i, line in enumerate(captain_prompt.split('\n')):
        if 'TIS Haiku' in line or 'OCS Gemini' in line:
            print(f"  Line {i}: {line[:80]}")


if __name__ == "__main__":
    main()
