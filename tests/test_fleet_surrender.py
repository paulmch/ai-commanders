"""
Fleet-battle surrenders are recorded facts, and the result reason says how a
fleet was lost.

Before: in a fleet battle a surrender was only visible as the captain's tool
call (no `surrender` event), and a fleet whose last ships struck their
colours ended as "<Side> fleet eliminated" with those ships at 100 % hull.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import battle_damage as bd
from src.combat import HitLocation
from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.fleet_config import BattleFleetConfig
from src.llm.victory import BattleOutcome


@pytest.fixture(scope="module")
def fleet():
    return load_fleet_data("data/fleet_ships.json")


def make_runner(fleet, alpha_types, beta_types, record=True):
    config = BattleFleetConfig.from_dict({
        "alpha_fleet": {"ships": [{"ship_type": t, "model": "heuristic"} for t in alpha_types]},
        "beta_fleet": {"ships": [{"ship_type": t, "model": "heuristic"} for t in beta_types]},
        "record_battle": record, "personality_selection": False, "seed": 5,
    })
    runner = LLMBattleRunner(
        BattleConfig.from_fleet_config(config, verbose=False),
        LLMCaptainConfig("A", "A"), LLMCaptainConfig("B", "B"), Mock(), config,
    )
    runner.setup_fleet_battle(fleet)
    return runner


def _disarm(sim, ship):
    for fitting in bd._gun_fittings(ship, None) + bd._launcher_fittings(ship):
        bd.damage_fitting(sim, ship, fitting, 100.0, "surface_hit", "x", HitLocation.LATERAL)


def _captain_surrenders(runner, ship_id, reason=None):
    captain = {**runner.alpha_captains, **runner.beta_captains}[ship_id]
    args = {"reason": reason} if reason else {}
    captain._execute_tool(ToolCall(id="t", name="surrender", arguments=args),
                          runner.simulation, ship_id)
    assert captain.has_surrendered
    return captain


def _surrenders(runner):
    return [e for e in runner.recorder.events if getattr(e.event_type, "value", e.event_type) == "surrender"]


# ---------------------------------------------------------------------------
# 1. The surrender event
# ---------------------------------------------------------------------------

def test_a_captains_fleet_surrender_is_recorded_once_with_its_facts(fleet):
    runner = make_runner(fleet, ["frigate"], ["corvette", "corvette"])
    sim = runner.simulation
    sim.current_time = 120.0
    beta_2 = runner.beta_ships["beta_2"]
    _disarm(sim, beta_2)
    captain = _captain_surrenders(runner, "beta_2", reason="all weapons out")
    runner._check_fleet_surrender_draw()
    runner._check_fleet_surrender_draw()          # next checkpoint: no second event
    events = _surrenders(runner)
    assert len(events) == 1
    e = events[0]
    assert (e.timestamp, e.ship_id) == (120.0, "beta_2")
    assert e.data == {"captain_name": captain.name, "faction": "beta",
                      "reason": "all weapons out", "hull_pct": 100.0,
                      "could_deal_damage": False}
    assert beta_2.is_surrendered


def test_could_deal_damage_is_read_before_the_ship_is_flagged(fleet):
    # An armed ship that gives up could still have fought; a surrendered
    # ship never can, so reading it after the flag would always say False.
    runner = make_runner(fleet, ["frigate"], ["corvette"])
    _captain_surrenders(runner, "beta_1")
    runner._check_fleet_surrender_draw()
    (e,) = _surrenders(runner)
    assert e.data["could_deal_damage"] is True
    assert e.data["reason"] is None               # the LLM tool takes no reason


def test_an_mcp_fleet_surrender_records_every_ship_still_afloat(fleet):
    runner = make_runner(fleet, ["frigate", "frigate", "frigate"], ["destroyer"])
    runner.alpha_ships["alpha_1"].is_destroyed = True
    runner.alpha_ships["alpha_2"].is_dying = True
    runner._surrender_fleet_ships(runner.alpha_ships, "Commander MCP")
    runner._surrender_fleet_ships(runner.alpha_ships, "Commander MCP")
    events = _surrenders(runner)
    assert [e.ship_id for e in events] == ["alpha_3"]   # dead ships do not surrender
    assert events[0].data["captain_name"] == "Commander MCP"
    assert all(s.is_surrendered for s in runner.alpha_ships.values())  # flag as before


def test_the_mcp_http_loop_records_the_fleet_surrender(fleet):
    from src.llm.mcp_http_server import apply_mcp_fleet_control
    runner = make_runner(fleet, ["frigate", "frigate"], ["destroyer"])
    runner.alpha_mcp = SimpleNamespace(name="Commander MCP", has_surrendered=True,
                                       has_proposed_draw=False, has_accepted_draw=False)
    assert apply_mcp_fleet_control(runner) is False
    assert [e.ship_id for e in _surrenders(runner)] == ["alpha_1", "alpha_2"]


# ---------------------------------------------------------------------------
# 2. The result reason says how the fleet was lost (outcome unchanged)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fates, reason", [
    (["destroyed", "destroyed", "dying"], "Beta fleet eliminated"),
    (["surrendered", "surrendered", "surrendered"],
     "Beta fleet surrendered (3 ships struck their colours)"),
    (["destroyed", "surrendered", "surrendered"],
     "Beta fleet destroyed or surrendered (1 destroyed, 2 surrendered)"),
])
def test_the_reason_names_how_the_fleet_was_lost(fleet, fates, reason):
    runner = make_runner(fleet, ["frigate"], ["corvette"] * 3, record=False)
    for ship, fate in zip(runner.beta_ships.values(), fates):
        setattr(ship, f"is_{fate}", True)
    # (a dying hulk keeps the battle running until it detonates)
    assert runner._is_fleet_battle_over() is ("dying" not in fates)
    result = runner._evaluate_fleet_result()
    assert result.outcome == BattleOutcome.ALPHA_VICTORY and result.winner == "alpha"
    assert result.reason == reason


def test_one_surrendered_ship_reads_in_the_singular(fleet):
    runner = make_runner(fleet, ["frigate"], ["corvette"], record=False)
    runner.beta_ships["beta_1"].is_surrendered = True
    assert runner._evaluate_fleet_result().reason == "Beta fleet surrendered (1 ship struck its colours)"


def test_an_mcp_fleet_surrender_does_not_count_its_dead_as_surrendered(fleet):
    runner = make_runner(fleet, ["frigate", "frigate"], ["destroyer"], record=False)
    runner.alpha_ships["alpha_1"].is_destroyed = True
    runner._surrender_fleet_ships(runner.alpha_ships, "Commander MCP")
    result = runner._evaluate_fleet_result()
    assert result.winner == "beta"
    assert result.reason == "Alpha fleet destroyed or surrendered (1 destroyed, 1 surrendered)"


def test_mutual_loss_names_both_fleets(fleet):
    runner = make_runner(fleet, ["frigate"], ["frigate"], record=False)
    runner.alpha_ships["alpha_1"].is_destroyed = True
    runner.beta_ships["beta_1"].is_surrendered = True
    result = runner._evaluate_fleet_result()
    assert result.outcome == BattleOutcome.DRAW and result.winner is None
    assert result.reason == ("Mutual destruction/surrender: Alpha fleet eliminated; "
                             "Beta fleet surrendered (1 ship struck its colours)")


def test_combat_ineffective_ending_keeps_its_reason(fleet):
    runner = make_runner(fleet, ["frigate"], ["frigate"], record=False)
    assert not runner._is_fleet_battle_over()
    _disarm(runner.simulation, runner.alpha_ships["alpha_1"])
    result = runner._evaluate_fleet_result()
    assert result.winner == "beta"
    assert result.reason.startswith("Alpha fleet combat-ineffective")
