"""
Pass-5 empirical hunt: defects found by running seeded heuristic battles
(scripts/run_draft_battle.py --auto-draft --no-admirals --trace) and checking
their recordings against the game's own rules. Each test fails on the engine
as it was before its fix.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.fleet_config import BattleFleetConfig
from src.llm.victory import BattleOutcome


@pytest.fixture(scope="module")
def fleet():
    return load_fleet_data("data/fleet_ships.json")


def make_runner(fleet, alpha_types, beta_types, **options):
    config = BattleFleetConfig.from_dict({
        "alpha_fleet": {"ships": [{"ship_type": t, "model": "heuristic"} for t in alpha_types]},
        "beta_fleet": {"ships": [{"ship_type": t, "model": "heuristic"} for t in beta_types]},
        "record_battle": False, "personality_selection": False,
        "seed": 5, **options,
    })
    runner = LLMBattleRunner(
        BattleConfig.from_fleet_config(config, verbose=False),
        LLMCaptainConfig("A", "A"), LLMCaptainConfig("B", "B"), Mock(), config,
    )
    runner.setup_fleet_battle(fleet)
    return runner


# ---------------------------------------------------------------------------
# 1. Time-limit verdict: every hull has a class value.
#
# 9 of 10 seeded heuristic battles ended on the time-limit "tactical
# advantage" path. Its class-value table had no entry for the torpedo cruiser
# or the siege dreadnought, so both silently scored the destroyer default
# (20): a dead siege dreadnought cost its fleet 30 points instead of 105.
# ---------------------------------------------------------------------------

def test_every_hull_type_has_a_tactical_class_value():
    from src.llm.battle_runner import SHIP_CLASS_VALUES, ship_class_value
    with open("data/fleet_ships.json") as f:
        hulls = json.load(f)["ships"]
    for ship_type in hulls:
        assert ship_type in SHIP_CLASS_VALUES, ship_type
    assert ship_class_value("cruiser_torpedo") == ship_class_value("cruiser")
    assert ship_class_value("dreadnought_siege") == ship_class_value("dreadnought")


def test_losing_a_siege_dreadnought_costs_as_much_as_a_dreadnought(fleet):
    # Mirror losses: each side loses its dreadnought-class hull. Before the
    # fix alpha "won" 125 vs 50 because its siege dreadnought was priced as
    # a destroyer.
    runner = make_runner(fleet, ["dreadnought_siege", "destroyer"],
                         ["dreadnought", "destroyer"])
    runner.alpha_ships["alpha_1"].is_destroyed = True
    runner.beta_ships["beta_1"].is_destroyed = True
    outcome, winner, reason = runner._evaluate_fleet_tactical_advantage()
    assert outcome == BattleOutcome.DRAW, reason
    assert winner is None


# ---------------------------------------------------------------------------
# 2. Draw-accepted points ignore lost ships.
#
# _evaluate_fleet_result summed _calculate_battle_points over EVERY ship, so
# a destroyed hull still earned its hull-integrity bonus (up to 50) and its
# delta-v bonus (up to 30): a fleet that had lost half its ships could win
# an agreed draw "on points".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["is_destroyed", "is_dying", "is_surrendered"])
def test_agreed_draw_does_not_score_lost_ships(fleet, status):
    runner = make_runner(fleet, ["destroyer"], ["destroyer", "destroyer"])
    runner.alpha_mcp = SimpleNamespace(has_proposed_draw=True, has_accepted_draw=False)
    runner.beta_admiral = SimpleNamespace(has_proposed_draw=False, has_accepted_draw=True)
    setattr(runner.beta_ships["beta_2"], status, True)
    result = runner._evaluate_fleet_result()
    assert "Draw accepted" in result.reason
    # Equal surviving hulls, beta lost one: beta must not come out ahead.
    assert result.outcome != BattleOutcome.BETA_VICTORY, result.reason


# ---------------------------------------------------------------------------
# 3. A torpedo that leaves the battle gets an outcome.
#
# _update_torpedoes culls a round 10,000 km from its target (or after an
# hour) with no event at all. In a seeded 900 s battle (seed 8, budget 200)
# 9 of 78 launched torpedoes - all seeker-killed rounds coasting away - simply
# vanished from the recording: no impact, no miss, no in-flight-at-end entry,
# so the torpedo ledger (launched = hit + missed + intercepted + in flight)
# could not balance. The cull now logs the round as a torpedo miss with
# reason 'left_battle'.
# ---------------------------------------------------------------------------

def _torpedo_sim(fleet_data_json):
    from src.physics import Vector3D
    from src.simulation import CombatSimulation, create_ship_from_fleet_data
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=3)
    for sid, faction, x in (("atk", "alpha", -1000.0), ("tgt", "beta", 0.0)):
        sim.add_ship(create_ship_from_fleet_data(
            ship_id=sid, ship_type="destroyer", faction=faction,
            fleet_data=fleet_data_json, position=Vector3D(x, 0, 0),
            velocity=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0)))
    return sim


def _inject_torpedo(sim, position, velocity, disabled):
    from src.simulation import TorpedoInFlight
    from src.torpedo import GuidanceMode, Torpedo, TorpedoSpecs
    torp = Torpedo(specs=TorpedoSpecs.trident(), position=position,
                   velocity=velocity, target_id="tgt",
                   guidance_mode=GuidanceMode.COLLISION)
    torp.armed = True
    if disabled:
        torp.fuel_exhausted = True
        torp.remaining_delta_v_kps = 0.0
    flight = TorpedoInFlight(torpedo_id="torp_gone", torpedo=torp,
                             source_ship_id="atk", launch_time=sim.current_time)
    flight.is_disabled = disabled
    sim.torpedoes.append(flight)
    return flight


@pytest.mark.parametrize("disabled", [True, False])
def test_torpedo_culled_at_battle_edge_is_logged_as_a_miss(disabled):
    from src.physics import Vector3D
    from src.simulation import SimulationEventType
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    sim = _torpedo_sim(data)
    # Coasting away from the target, just past the 10,000 km cull line.
    _inject_torpedo(sim, Vector3D(10_000_500.0, 0, 0), Vector3D(14_000.0, 0, 0), disabled)
    sim.step()
    assert not sim.torpedoes
    misses = [e for e in sim.events
              if e.event_type == SimulationEventType.PROJECTILE_MISS
              and e.data.get("type") == "torpedo"]
    assert len(misses) == 1
    assert misses[0].data["projectile_id"] == "torp_gone"
    assert misses[0].data["reason"] == "left_battle"
    assert misses[0].target_id == "tgt"

