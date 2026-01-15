#!/usr/bin/env python3
"""Test to see what enemy ship names/IDs look like in tactical status."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.battle_runner import load_fleet_data
from src.simulation import CombatSimulation, create_ship_from_fleet_data
from src.physics import Vector3D

def main():
    fleet_data = load_fleet_data()
    sim = CombatSimulation()

    # Add alpha ship
    alpha_1 = create_ship_from_fleet_data(
        ship_id="alpha_1",
        ship_type="destroyer",
        faction="alpha",
        fleet_data=fleet_data,
        position=Vector3D(-200000, 0, 0),
        velocity=Vector3D(2500, 0, 0),
        forward=Vector3D(1, 0, 0),
    )
    alpha_1.name = "TIS Haiku-1"
    sim.add_ship(alpha_1)

    # Add beta ship
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

    print("=" * 60)
    print("SHIP DATA IN SIMULATION")
    print("=" * 60)
    for ship_id, ship in sim.ships.items():
        print(f"\nship_id: {ship_id}")
        print(f"  ship.name: {getattr(ship, 'name', 'NO NAME ATTR')}")
        print(f"  ship.ship_id: {ship.ship_id}")

    print("\n" + "=" * 60)
    print("ENEMY SHIPS from alpha_1's perspective")
    print("=" * 60)
    enemies = sim.get_enemy_ships("alpha_1")
    for e in enemies:
        print(f"\nEnemy ship_id: {e.ship_id}")
        print(f"  Enemy name: {getattr(e, 'name', 'NO NAME ATTR')}")
        print(f"  hasattr(name): {hasattr(e, 'name')}")

    print("\n" + "=" * 60)
    print("What _build_enemy_info would produce:")
    print("=" * 60)
    for e in enemies:
        info = {
            "ship_id": e.ship_id,
            "name": getattr(e, 'name', e.ship_id),
        }
        print(f"  ship_id: {info['ship_id']}")
        print(f"  name: {info['name']}")

if __name__ == "__main__":
    main()
