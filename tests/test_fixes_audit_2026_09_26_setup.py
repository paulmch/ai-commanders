"""Offline regression probes for the setup/damage audit of 2026-09-26."""

import asyncio
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.damage import DamageCone, DamagePropagator, Module, ModuleLayout
from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.fleet_config import AdmiralConfig, BattleFleetConfig, DraftConfig, validate_fleet_config
from src.llm.fleet_draft import (
    FORMATION_MAX_OFFSET_KM, MIN_SEPARATION_KM, SHIP_POINT_COSTS,
    apply_formation, auto_draft, name_drafted_ships, validate_selection,
)
from src.llm.mcp_controller import apply_mcp_commands_to_simulation
from src.llm.mcp_state import MCPCommand, MCPCommandType
from src.llm.victory import BattleOutcome
from src.modules import ModuleLayout as ShipModuleLayout
from src.physics import Vector3D
from src.thermal import RadiatorState


@pytest.fixture(scope="module")
def fleet():
    return load_fleet_data()


def make_runner(fleet, *, record=False, **options):
    config = BattleFleetConfig.from_dict({
        "alpha_fleet": {"ships": [{"ship_type": "destroyer", "model": "heuristic"}]},
        "beta_fleet": {"ships": [{"ship_type": "destroyer", "model": "heuristic"}]},
        "record_battle": record, "personality_selection": False,
        "seed": 13, **options,
    })
    runner = LLMBattleRunner(
        BattleConfig.from_fleet_config(config, verbose=False),
        LLMCaptainConfig("A", "A"), LLMCaptainConfig("B", "B"), Mock(), config,
    )
    runner.setup_fleet_battle(fleet)
    return runner


@pytest.mark.parametrize("faction", ["alpha", "beta"])
def test_disabled_admiral_stays_disabled(fleet, faction):
    runner = make_runner(fleet, **{f"{faction}_fleet": {
        "admiral": {"model": "unused", "enabled": False},
        "ships": [{"ship_type": "destroyer", "model": "heuristic"}],
    }})
    assert getattr(runner, faction + "_admiral") is None
    assert not getattr(runner, faction + "_captains")[faction + "_1"].has_admiral


def test_heuristic_retracts_real_extended_radiators_when_cool(fleet):
    runner = make_runner(fleet)
    ship = runner.alpha_ships["alpha_1"]
    ship.thermal_system.radiators.extend_all()
    captain = runner.alpha_captains[ship.ship_id]
    for command in captain.decide(ship.ship_id, runner.simulation):
        runner.simulation.inject_command(ship.ship_id, command)
    assert all(r.state != RadiatorState.EXTENDED
               for r in ship.thermal_system.radiators.radiators.values())


@pytest.mark.parametrize("ys", [[150.0] * 8, [0.0, 4.0, -4.0, -4.0]])
def test_formation_nudges_preserve_bounds_and_all_separations(ys):
    ships = name_drafted_ships(["frigate"] * len(ys), "alpha")
    apply_formation(ships, [{"ship_name": s.ship_name, "y_km": y}
                            for s, y in zip(ships, ys)])
    for i, ship in enumerate(ships):
        assert all(abs(c) <= FORMATION_MAX_OFFSET_KM for c in ship.offset_km)
        for other in ships[:i]:
            assert math.dist(ship.offset_km, other.offset_km) >= MIN_SEPARATION_KM


@pytest.mark.parametrize("payload", [[None], [{"ship_type": "frigate", "count": True}],
                                    [{"ship_type": "frigate", "count": 10 ** 100}]])
def test_invalid_draft_payload_is_rejected_without_expanding_it(payload):
    flat, spent, error = validate_selection(payload, budget=100, max_ships=8)
    assert flat is None and spent == 0 and error


def test_over_budget_error_still_reports_fleet_cost():
    """Drafting LLMs/MCP clients need the total to correct an illegal pick."""
    cost = SHIP_POINT_COSTS["dreadnought"] * 2
    _, _, error = validate_selection([{"ship_type": "dreadnought", "count": 2}],
                                     budget=cost - 1, max_ships=8)
    assert f"costs {cost} points" in error


@pytest.mark.parametrize("budget,max_ships", [(0, 8), (100, 0)])
def test_impossible_auto_draft_is_rejected(budget, max_ships):
    with pytest.raises(ValueError):
        auto_draft("alpha", budget=budget, max_ships=max_ships)


@pytest.mark.parametrize("variant,base", [("cruiser_torpedo", "cruiser"),
                                         ("dreadnought_siege", "dreadnought")])
def test_variant_uses_its_hull_class_damage_layout(fleet, variant, base):
    layout = ShipModuleLayout.from_ship_type(variant, fleet)
    base_layout = ShipModuleLayout.from_ship_type(base, fleet)
    assert layout.total_layers == base_layout.total_layers
    for expected in base_layout.get_critical_modules():
        actual = layout.get_module_by_name(expected.name)
        assert actual.position == expected.position
        assert actual.armor_rating == expected.armor_rating


def test_spalling_cannot_create_energy():
    layout = ModuleLayout(modules=[
        Module("fragile", Vector3D(1, 0, 0), 0.1, 0.1, radius_m=2),
        Module("sink", Vector3D(2, 0, 0), 100, 100, radius_m=2),
    ])
    cone = DamageCone.from_weapon_type(Vector3D.zero(), Vector3D(1, 0, 0), 10)
    results = DamagePropagator().propagate(cone, layout)
    assert sum(r.damage_taken_gj for r in results) <= 10


@pytest.mark.parametrize("command_type,params", [
    (MCPCommandType.SET_MANEUVER, {"maneuver_type": "BRAKE"}),
    (MCPCommandType.SET_PRIMARY_TARGET, {"target_id": "alpha_1"}),
    (MCPCommandType.SET_RADIATORS, {"extend": True}),
    (MCPCommandType.SET_WEAPONS_ORDER, {"spinal_mode": "FIRE_IMMEDIATE"}),
])
def test_mcp_cannot_command_opponent(fleet, command_type, params):
    runner = make_runner(fleet)
    command = MCPCommand(command_type, ship_id="beta_1", parameters=params)
    result = apply_mcp_commands_to_simulation([command], runner.simulation, "alpha")
    assert result["errors"] and not result["applied"]
    assert runner.beta_ships["beta_1"].primary_target_id is None
    assert runner.beta_ships["beta_1"].current_maneuver is None
    assert not runner.beta_ships["beta_1"].weapons_orders


@pytest.mark.parametrize("status", ["is_destroyed", "is_dying", "is_surrendered"])
def test_mcp_rejects_commands_for_inactive_ships(fleet, status):
    runner = make_runner(fleet)
    ship = runner.alpha_ships["alpha_1"]
    setattr(ship, status, True)
    commands = [MCPCommand(MCPCommandType.SET_MANEUVER, ship_id=ship.ship_id,
                           parameters={"maneuver_type": "BRAKE"}),
                MCPCommand(MCPCommandType.SET_RADIATORS, ship_id=ship.ship_id,
                           parameters={"extend": True})]
    result = apply_mcp_commands_to_simulation(commands, runner.simulation, "alpha")
    assert len(result["errors"]) == 2 and not result["applied"]


def test_mixed_mcp_admiral_draw_ends_battle(fleet):
    runner = make_runner(fleet)
    runner.alpha_mcp = SimpleNamespace(has_proposed_draw=True, has_accepted_draw=False)
    runner.beta_admiral = SimpleNamespace(has_proposed_draw=False, has_accepted_draw=True)
    assert runner._is_fleet_battle_over()
    result = runner._evaluate_fleet_result()
    assert "Draw accepted" in result.reason


def test_dying_fleet_is_not_reported_as_surviving_at_time_limit(fleet):
    runner = make_runner(fleet)
    runner.alpha_ships["alpha_1"].is_dying = True
    runner.simulation.current_time = runner.config.time_limit_s
    result = runner._evaluate_fleet_result()
    assert result.outcome == BattleOutcome.BETA_VICTORY
    assert result.reason == "Alpha fleet eliminated"


def test_fleet_recording_preserves_unlimited_mode(fleet):
    runner = make_runner(fleet, record=True, unlimited_mode=True)
    assert runner.recorder.recording.unlimited_mode is True


def test_recorded_ship_specs_match_actual_weapon_and_acceleration(fleet):
    runner = make_runner(fleet)
    ship = runner.alpha_ships["alpha_1"]
    from src.llm.battle_recorder import BattleRecorder
    specs = BattleRecorder()._extract_ship_specs(ship)
    for slot, state in ship.weapons.items():
        assert specs["weapons"][slot]["damage_gj"] == state.weapon.kinetic_energy_gj
    assert specs["max_acceleration_g"] == pytest.approx(ship.kinematic_state.max_acceleration_g())


def test_checked_in_fleet_configs_have_valid_ships_and_vectors(fleet):
    root = Path(__file__).resolve().parents[1]
    for path in (root / "data").glob("fleet_config_*.json"):
        config = BattleFleetConfig.from_json(str(path))
        assert not validate_fleet_config(config, fleet), path.name


def test_programmatic_disabled_admiral_stays_disabled(fleet):
    runner = make_runner(fleet)
    runner.fleet_config.alpha_fleet.admiral = AdmiralConfig(model="unused", enabled=False)
    runner.setup_fleet_battle(fleet)
    assert runner.alpha_admiral is None
    assert not runner.alpha_captains["alpha_1"].has_admiral


@pytest.mark.parametrize("options", [{"budget": 0}, {"max_ships": 0}, {"budget": True}])
def test_impossible_draft_config_rejected(options):
    with pytest.raises(ValueError):
        DraftConfig(**options)


def test_auto_draft_can_buy_cheapest_hull_under_any_doctrine():
    budget = min(SHIP_POINT_COSTS.values())
    for seed in range(12):
        draft = auto_draft("alpha", budget=budget, seed=seed)
        assert draft.ships and draft.points_spent <= budget


def test_spalling_uses_one_budget_independent_of_layout_insertion_order():
    def run(reverse):
        modules = [Module("origin", Vector3D(1, 0, 0), 0.1, 0.1, radius_m=2)]
        # Outside the narrow primary cone, inside the secondary spall cone.
        modules += [Module(f"side{i}", Vector3D(2 + i, 1 + i / 2, 0), 100, 100,
                           radius_m=10) for i in range(3)]
        layout = ModuleLayout(modules=list(reversed(modules)) if reverse else modules)
        cone = DamageCone.from_weapon_type(Vector3D.zero(), Vector3D(1, 0, 0), 10)
        results = DamagePropagator().propagate(cone, layout)
        return {m.name: m.health for m in modules}, sum(r.damage_taken_gj for r in results)
    forward, total = run(False)
    reverse, reverse_total = run(True)
    assert forward == reverse
    assert total <= 10 and reverse_total <= 10


def test_http_runner_honors_fractional_checkpoints_and_time_limit(fleet):
    from src.llm.mcp_http_server import run_fleet_battle_with_http
    runner = make_runner(fleet, decision_interval_s=20.5, time_limit_s=41.25)
    result = asyncio.run(run_fleet_battle_with_http(runner, fleet, Mock()))
    assert result.duration_s == pytest.approx(41.25)
    assert [entry["time"] for entry in result.decision_log] == [20.5, 41.0]


def test_mcp_draw_proposal_is_passed_to_opponent_admiral(fleet):
    runner = make_runner(fleet)
    runner.alpha_mcp = SimpleNamespace(has_proposed_draw=True, has_accepted_draw=False)
    admiral = Mock(faction="beta")
    runner.beta_admiral = admiral
    runner._get_admiral_decision(admiral, [], None)
    assert admiral.decide.call_args.kwargs["enemy_admiral"] is runner.alpha_mcp


def test_mcp_only_cli_does_not_construct_api_client(tmp_path, monkeypatch):
    import scripts.mcp_battle as cli
    import src.llm.mcp_http_server as server
    config = {
        f"{side}_fleet": {"mcp": {"enabled": True}, "ships": [
            {"ship_type": "frigate", "model": "mcp"}]} for side in ("alpha", "beta")
    }
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))
    client = Mock(side_effect=AssertionError("Offline MCP battle must not need an API key"))
    monkeypatch.setattr(cli, "CaptainClient", client)

    async def local_battle(*args, **kwargs):
        return SimpleNamespace(outcome=BattleOutcome.DRAW, winner=None, reason="stub",
                               duration_s=0, checkpoints_used=0)

    monkeypatch.setattr(server, "run_battle_with_http_server", local_battle)
    asyncio.run(cli.run_mcp_battle(str(path), verbose=False))
    client.assert_not_called()


def test_draft_cli_captain_override_initializes_client(monkeypatch):
    import scripts.run_draft_battle as cli
    import src.llm.client as client_module
    client = Mock()
    monkeypatch.setattr(client_module, "CaptainClient", client)
    runner = Mock()
    runner.return_value.run_battle.return_value = SimpleNamespace(
        winner=None, reason="stub", duration_s=0, checkpoints_used=0, recording_file=None)
    monkeypatch.setattr(cli, "LLMBattleRunner", runner)
    monkeypatch.setattr(sys, "argv", ["run_draft_battle.py", "--auto-draft", "--no-admirals",
                                      "--alpha-captain-model", "test/model", "--quiet"])
    assert cli.main() == 0
    client.assert_called_once()
    assert runner.call_args.kwargs["client"] is client.return_value


@pytest.mark.parametrize("check", ["heat_capacity", "dying"])
def test_mcp_state_reports_actual_heat_capacity_and_active_ships(fleet, check):
    from src.llm.mcp_controller import MCPController, MCPControllerConfig
    runner = make_runner(fleet)
    controller = MCPController(MCPControllerConfig(faction="alpha"), fleet)
    captain = runner.alpha_captains["alpha_1"]
    ship = runner.alpha_ships["alpha_1"]
    if check == "heat_capacity":
        state = controller.build_state_for_mcp(runner.simulation, [captain])
        assert state.friendly_ships[0]["heatsink_capacity_gj"] == ship.thermal_system.heatsink.capacity_gj
    else:
        ship.is_dying = True
        assert not controller.build_state_for_mcp(runner.simulation, [captain]).friendly_ships


def test_heuristic_dying_captain_cannot_issue_orders(fleet):
    runner = make_runner(fleet)
    runner.alpha_ships["alpha_1"].is_dying = True
    assert runner.alpha_captains["alpha_1"].decide("alpha_1", runner.simulation) == []


def test_duel_victory_counts_dying_hulk_as_killed(fleet):
    runner = make_runner(fleet)
    runner.alpha_ships["alpha_1"].is_dying = True
    outcome, winner, _ = runner.evaluator.evaluate(
        runner.alpha_ships["alpha_1"], runner.beta_ships["beta_1"], at_time_limit=True)
    assert outcome == BattleOutcome.BETA_VICTORY and winner == "beta"


@pytest.mark.parametrize("check", ["siege", "salvo"])
def test_ship_document_covers_siege_variant_and_actual_salvo(fleet, check):
    from src.llm.tools import get_captain_tools_for_ship
    docs = (Path(__file__).resolve().parents[1] / "docs/ships.md").read_text()
    if check == "siege":
        assert "| Siege Dreadnought |" in docs
        return
    tools = get_captain_tools_for_ship("cruiser_torpedo", fleet, has_torpedoes=True)
    launch = next(t for t in tools if t["function"]["name"] == "launch_torpedo")
    count = launch["function"]["parameters"]["properties"]["count"]["maximum"]
    assert f"fire {count} torpedoes per 30s decision cycle" in docs
