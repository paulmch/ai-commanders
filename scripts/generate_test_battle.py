#!/usr/bin/env python3
"""
Generate battle recordings with scripted (non-LLM) captains.

The replay viewer needs recordings that exercise every mechanic the sim can
produce - torpedo salvos, PD dwell, coilgun exchanges, evasion, destruction -
without paying for LLM calls or hoping two captains improvise the right
battle. Each scenario here is a fixed per-checkpoint list of the same tool
calls a real captain would emit, executed through LLMCaptain._execute_tool so
the command path (and therefore the recording) is identical to a live battle.

Usage:
    uv run python scripts/generate_test_battle.py torpedo_strike
    uv run python scripts/generate_test_battle.py gun_duel
    uv run python scripts/generate_test_battle.py all
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.prompts import CaptainPersonality
from src.llm.battle_runner import LLMBattleRunner, BattleConfig, load_fleet_data

# A script maps checkpoint number (1-based, the value decide() sees as
# decision_count + 1) to the tool calls to issue at that checkpoint.
Script = Dict[int, List[Tuple[str, Dict[str, Any]]]]


class ScriptedCaptain(LLMCaptain):
    """An LLMCaptain whose decisions come from a fixed script, not an API."""

    def __init__(self, config: LLMCaptainConfig, script: Script):
        super().__init__(config, client=None)  # client is never used
        self.script = script

    def select_personality(self, distance_km: float, verbose: bool = False) -> Dict[str, Any]:
        return {}

    def decide(self, ship_id: str, simulation: Any) -> List[Any]:
        if self.has_surrendered:
            return []
        ship = simulation.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return []

        # Same lazy weapon-group setup as the real decide(); set_weapons_order
        # silently matches nothing without it.
        if not self.weapon_groups and self.config.fleet_data and self.config.ship_type:
            self.setup_weapon_groups(self.config.ship_type, self.config.fleet_data)

        checkpoint = self.decision_count + 1
        calls = self.script.get(checkpoint, [])

        commands: List[Any] = []
        tool_calls = [
            ToolCall(id=f"scripted_{checkpoint}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ]
        for tc in tool_calls:
            cmd = self._execute_tool(tc, simulation, ship_id)
            if cmd is not None:
                if isinstance(cmd, list):
                    commands.extend(c for c in cmd if c is not None)
                else:
                    commands.append(cmd)

        self.decision_count += 1
        self.last_tool_calls = tool_calls
        self.decision_history.append({
            "checkpoint": self.decision_count,
            "time": simulation.current_time,
            "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in tool_calls],
            "commands_count": len(commands),
        })
        return commands


class ScriptedBattleRunner(LLMBattleRunner):
    """Battle runner that swaps in scripted captains after normal setup."""

    def __init__(self, *args, alpha_script: Script, beta_script: Script, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha_script = alpha_script
        self._beta_script = beta_script

    def setup_battle(self, fleet_data: Dict[str, Any]) -> None:
        super().setup_battle(fleet_data)
        self.alpha_captain = ScriptedCaptain(self.alpha_config, self._alpha_script)
        self.beta_captain = ScriptedCaptain(self.beta_config, self._beta_script)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def torpedo_strike() -> Tuple[BattleConfig, LLMCaptainConfig, LLMCaptainConfig, Script, Script]:
    """
    Two-sided torpedo exchange: a 4-PD torpedo cruiser against an agile
    corvette. The corvette's launches fly into massed PD (3+ turrets is the
    doctrine threshold for blinding a seeker) -> intercepted / disabled
    torpedoes; the cruiser's launches chase a permanently evading 3g corvette
    -> misses and fuel exhaustion, with enough saturation late to land hits.

    Doctrine note: EVADE lasts one 30s decision window, so it must be
    re-issued every checkpoint to stay in effect.
    """
    config = BattleConfig(
        initial_distance_km=500.0,
        time_limit_s=900.0,
        max_checkpoints=20,
        alpha_ship_type="cruiser_torpedo",
        beta_ship_type="corvette",
        verbose=True,
        personality_selection=False,
        record_battle=True,
        record_sim_trace=True,
        seed=42,
    )
    alpha_cfg = LLMCaptainConfig(
        name="Captain Script-A",
        ship_name="TIS Harpoon",
        model="scripted-torpedo",
        personality=CaptainPersonality.BALANCED,
    )
    beta_cfg = LLMCaptainConfig(
        name="Captain Script-B",
        ship_name="HFS Dartfish",
        model="scripted-defender",
        personality=CaptainPersonality.BALANCED,
    )

    # Alpha keeps thrusting the whole time so blinded corvette rounds - whose
    # guidance froze on a stale solution - miss and chase into fuel
    # exhaustion. The launch ramp starts as single probing rounds (the
    # corvette's evasion and PD heat get a chance to matter) and only
    # saturates late, so the finale still lands.
    alpha_script: Script = {
        1: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.7}),
            ("launch_torpedo", {"count": 1}),
            ("send_message", {"message": "Tube one away. Your move, Dartfish."})],
        2: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.7}),
            ("launch_torpedo", {"count": 1})],
        3: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 0.8}),
            ("launch_torpedo", {"count": 2})],
        4: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 0.8}),
            ("launch_torpedo", {"count": 2})],
        5: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.7}),
            ("launch_torpedo", {"count": 2})],
        6: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 0.8}),
            ("launch_torpedo", {"count": 2})],
        7: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}),
            ("launch_torpedo", {"count": 2})],
        8: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 0.8}),
            ("launch_torpedo", {"count": 4})],
        9: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}),
            ("launch_torpedo", {"count": 4})],
        10: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}),
             ("launch_torpedo", {"count": 4})],
        11: [("launch_torpedo", {"count": 4})],
        12: [("launch_torpedo", {"count": 4})],
        13: [("launch_torpedo", {"count": 4})],
    }
    # The corvette never stops dancing, and spends its whole 8-round magazine
    # early so its torpedoes are in flight while the cruiser's PD is fresh.
    beta_script: Script = {
        cp: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0})]
        for cp in range(1, 21)
    }
    beta_script[1] = [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}),
                      ("launch_torpedo", {"count": 2}),
                      ("set_radiators", {"extend": True})]
    beta_script[2] = [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}),
                      ("launch_torpedo", {"count": 2}),
                      ("send_message", {"message": "Nice fireworks. Watch mine fly."})]
    beta_script[3] = [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}),
                      ("launch_torpedo", {"count": 2})]
    beta_script[4] = [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}),
                      ("launch_torpedo", {"count": 2})]
    return config, alpha_cfg, beta_cfg, alpha_script, beta_script


def gun_duel() -> Tuple[BattleConfig, LLMCaptainConfig, LLMCaptainConfig, Script, Script]:
    """
    Two destroyers close and slug it out with coilguns. Exercises:
    shot_fired / hit / miss, pd_fired dwell on slugs, pd_slug_damaged /
    destroyed, armor_damage, penetration, module damage, destruction.
    """
    config = BattleConfig(
        initial_distance_km=350.0,
        time_limit_s=900.0,
        max_checkpoints=24,
        alpha_ship_type="destroyer",
        beta_ship_type="destroyer",
        verbose=True,
        personality_selection=False,
        record_battle=True,
        record_sim_trace=True,
        seed=7,
    )
    alpha_cfg = LLMCaptainConfig(
        name="Captain Script-A",
        ship_name="TIS Broadside",
        model="scripted-gunner",
        personality=CaptainPersonality.BALANCED,
    )
    beta_cfg = LLMCaptainConfig(
        name="Captain Script-B",
        ship_name="HFS Riposte",
        model="scripted-gunner",
        personality=CaptainPersonality.BALANCED,
    )

    guns_free = ("set_weapons_order", {"spinal_mode": "FIRE_WHEN_OPTIMAL",
                                       "turret_mode": "FIRE_WHEN_OPTIMAL",
                                       "spinal_min_probability": 0.2,
                                       "turret_min_probability": 0.2})
    alpha_script: Script = {
        1: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}), guns_free,
            ("send_message", {"message": "Riposte, this is Broadside. Guns free."})],
        3: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0})],
        4: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.6})],
        6: [("set_maneuver", {"maneuver_type": "PADLOCK"})],
        9: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0})],
    }
    beta_script: Script = {
        1: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0}), guns_free],
        2: [("set_radiators", {"extend": True})],
        4: [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0})],
        5: [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 1.0})],
        8: [("set_maneuver", {"maneuver_type": "PADLOCK"})],
    }
    return config, alpha_cfg, beta_cfg, alpha_script, beta_script


SCENARIOS = {
    "torpedo_strike": torpedo_strike,
    "gun_duel": gun_duel,
}


def run_scenario(name: str) -> Optional[str]:
    config, alpha_cfg, beta_cfg, alpha_script, beta_script = SCENARIOS[name]()
    fleet_data = load_fleet_data()

    runner = ScriptedBattleRunner(
        config=config,
        alpha_config=alpha_cfg,
        beta_config=beta_cfg,
        client=None,  # scripted captains never call an LLM
        alpha_script=alpha_script,
        beta_script=beta_script,
    )
    result = runner.run_battle(fleet_data)

    print(f"\n=== {name} ===")
    print(f"Winner: {result.winner or 'Draw'} - {result.reason}")
    print(f"Duration: {result.duration_s:.0f}s, checkpoints: {result.checkpoints_used}")
    print(f"Recording: {result.recording_file}")
    return result.recording_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=[*SCENARIOS, "all"])
    args = parser.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        run_scenario(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
