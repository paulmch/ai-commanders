#!/usr/bin/env python3
"""Test script to see what the Admiral prompt looks like."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.admiral import (
    AdmiralSnapshot,
    FriendlyShipSnapshot,
    EnemyShipSnapshot,
    ProjectileSnapshot,
)
from src.llm.prompts import build_admiral_prompt
from src.llm.battle_runner import load_fleet_data


def main():
    fleet_data = load_fleet_data()

    # Create mock friendly ships
    friendly_ships = [
        FriendlyShipSnapshot(
            ship_id="alpha_1",
            ship_name="TIS Haiku-1",
            ship_type="destroyer",
            captain_name="Captain Haiku-1",
            position_km={"x": -200, "y": 0, "z": 0},
            velocity_kps=2.5,
            velocity_vector={"x": 1.0, "y": 0, "z": 0},
            hull_integrity=100.0,
            delta_v_remaining=450.0,
            heat_percent=15.0,
            max_acceleration_g=2.0,
            max_delta_v=500.0,
            weapons_summary="1x Spinal Coiler (4.3 GJ), 2x Coilgun (0.7 GJ)",
            weapons_ready=["Spinal Coiler Mk3", "Coilgun Mk2"],
            weapons_cooling=[],
            current_maneuver="INTERCEPT",
            current_target="OCS Gemini-1",
            radiators_extended=True,
            targeted_by=["OCS Gemini-1", "OCS Gemini-2"],
        ),
        FriendlyShipSnapshot(
            ship_id="alpha_2",
            ship_name="TIS Haiku-2",
            ship_type="destroyer",
            captain_name="Captain Haiku-2",
            position_km={"x": -200, "y": 5, "z": 0},
            velocity_kps=2.3,
            velocity_vector={"x": 1.0, "y": 0, "z": 0},
            hull_integrity=95.0,
            delta_v_remaining=440.0,
            heat_percent=20.0,
            max_acceleration_g=2.0,
            max_delta_v=500.0,
            weapons_summary="1x Spinal Coiler (4.3 GJ), 2x Coilgun (0.7 GJ)",
            weapons_ready=["Coilgun Mk2"],
            weapons_cooling=["Spinal Coiler Mk3"],
            current_maneuver="INTERCEPT",
            current_target="OCS Gemini-2",
            radiators_extended=True,
            targeted_by=[],
        ),
        FriendlyShipSnapshot(
            ship_id="alpha_3",
            ship_name="TIS Haiku-3",
            ship_type="dreadnought",
            captain_name="Captain Haiku-3",
            position_km={"x": -200, "y": 10, "z": 0},
            velocity_kps=1.5,
            velocity_vector={"x": 0.8, "y": 0, "z": 0},
            hull_integrity=100.0,
            delta_v_remaining=380.0,
            heat_percent=10.0,
            max_acceleration_g=0.75,
            max_delta_v=400.0,
            weapons_summary="2x Spinal Coiler (8.6 GJ), 4x Coilgun (2.8 GJ)",
            weapons_ready=["Spinal Coiler Mk3", "Coilgun Mk2"],
            weapons_cooling=[],
            current_maneuver="INTERCEPT",
            current_target="OCS Gemini-3",
            radiators_extended=True,
            targeted_by=["OCS Gemini-3"],
        ),
    ]

    # Create mock enemy ships
    enemy_ships = [
        EnemyShipSnapshot(
            ship_id="beta_1",
            ship_name="OCS Gemini-1",
            ship_type="destroyer",
            position_km={"x": 200, "y": 0, "z": 0},
            velocity_kps=2.4,
            velocity_vector={"x": -1.0, "y": 0, "z": 0},
            distance_from_closest_friendly_km=400.0,
            closing_rate_kps=4.9,
        ),
        EnemyShipSnapshot(
            ship_id="beta_2",
            ship_name="OCS Gemini-2",
            ship_type="destroyer",
            position_km={"x": 200, "y": 5, "z": 0},
            velocity_kps=2.2,
            velocity_vector={"x": -1.0, "y": 0, "z": 0},
            distance_from_closest_friendly_km=400.0,
            closing_rate_kps=4.5,
        ),
        EnemyShipSnapshot(
            ship_id="beta_3",
            ship_name="OCS Gemini-3",
            ship_type="dreadnought",
            position_km={"x": 200, "y": 10, "z": 0},
            velocity_kps=1.2,
            velocity_vector={"x": -0.8, "y": 0, "z": 0},
            distance_from_closest_friendly_km=400.0,
            closing_rate_kps=2.3,
        ),
    ]

    # Create snapshots
    snapshot_t_minus_15 = AdmiralSnapshot(
        timestamp=15.0,
        friendly_ships=friendly_ships,
        enemy_ships=enemy_ships,
        projectiles=[],
        fleet_summary="3 ships: 2 Destroyers, 1 Dreadnought",
    )

    snapshot_t_zero = AdmiralSnapshot(
        timestamp=30.0,
        friendly_ships=friendly_ships,
        enemy_ships=enemy_ships,
        projectiles=[],
        fleet_summary="3 ships: 2 Destroyers, 1 Dreadnought",
    )

    # Build the prompt
    prompt = build_admiral_prompt(
        admiral_name="Admiral Sonnet",
        faction="alpha",
        snapshot_t_minus_15=snapshot_t_minus_15,
        snapshot_t_zero=snapshot_t_zero,
        personality="I am a bold and decisive commander who believes in overwhelming force.",
        fleet_data=fleet_data,
        enemy_has_admiral=True,
        enemy_proposed_draw=False,
        received_messages=None,
        communications_log=None,
    )

    print("=" * 80)
    print("ADMIRAL PROMPT")
    print("=" * 80)
    print(prompt)
    print("=" * 80)


if __name__ == "__main__":
    main()
