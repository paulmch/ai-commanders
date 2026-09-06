"""End-to-end regressions for battle options, commander orders and ballistics.

All commander responses are local stubs; these tests never call a model API.
"""

import math
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.firecontrol import WeaponsCommand, WeaponsOrder, calculate_hit_probability
from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.fleet_config import BattleFleetConfig
from src.physics import (Vector3D, ShipState, intercept_time, propagate_state,
                         calculate_torque_from_thrust, MAX_GIMBAL_ANGLE_DEG)
from src.projectile import KineticProjectile
from src.simulation import (CombatSimulation, Maneuver, ManeuverType, ProjectileInFlight,
                            SimulationEventType, create_ship_from_fleet_data)
from src.targeting import LeadCalculator


@pytest.fixture(scope="module")
def fleet():
    return load_fleet_data()


def setup_ships(fleet, ship_type="destroyer", interval=30):
    sim = CombatSimulation(seed=19, decision_interval=interval)
    for ship_id, faction, x in (("a", "alpha", 0), ("b", "beta", 100_000), ("c", "beta", 200_000)):
        ship = create_ship_from_fleet_data(
            ship_id, ship_type, faction, fleet,
            position=Vector3D(x, 0, 0), forward=Vector3D(1 if faction == "alpha" else -1, 0, 0),
        )
        ship.name = ship_id
        sim.add_ship(ship)
    captain = LLMCaptain(LLMCaptainConfig(
        name="Captain", ship_name="a", ship_type=ship_type, fleet_data=fleet,
    ), Mock())
    captain.setup_weapon_groups(ship_type, fleet)
    return sim, captain


def execute(captain, sim, name, **args):
    command = captain._execute_tool(ToolCall(id="test", name=name, arguments=args), sim, "a")
    if command is not None:
        for item in command if isinstance(command, list) else [command]:
            assert sim.inject_command("a", item)
    return command


def fleet_dict(**options):
    return {
        "alpha_fleet": {"ships": [{"model": "heuristic", "ship_type": "destroyer"}]},
        "beta_fleet": {"ships": [{"model": "heuristic", "ship_type": "destroyer"}]},
        **options,
    }


def test_partial_position_survives_json_loading(fleet):
    data = fleet_dict(initial_distance_km=800)
    data["alpha_fleet"]["ships"][0]["position"] = {"y": 12}
    config = BattleFleetConfig.from_dict(data)
    runner = LLMBattleRunner(BattleConfig(record_battle=False, verbose=False),
                            LLMCaptainConfig("A", "A"), LLMCaptainConfig("B", "B"), Mock(), config)
    runner.setup_fleet_battle(fleet)
    assert runner.alpha_ships["alpha_1"].position.to_tuple() == (-400_000, 12_000, 0)


def test_duplicate_ship_ids_rejected_before_setup(fleet):
    data = fleet_dict()
    for faction in ("alpha", "beta"):
        data[f"{faction}_fleet"]["ships"][0]["ship_id"] = "duplicate"
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        config = BattleFleetConfig.from_dict(data)
        LLMBattleRunner(BattleConfig(record_battle=False), LLMCaptainConfig("A", "A"),
                        LLMCaptainConfig("B", "B"), Mock(), config).setup_fleet_battle(fleet)


@pytest.mark.parametrize("field,value", [
    ("decision_interval_s", 0), ("decision_interval_s", -1),
    ("decision_interval_s", float("nan")), ("time_limit_s", float("inf")),
    ("initial_distance_km", -10),
])
def test_invalid_battle_options_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        BattleFleetConfig.from_dict(fleet_dict(**{field: value}))


@pytest.mark.parametrize("interval", [20, 45, 60])
@pytest.mark.parametrize("tool,args", [
    ("set_maneuver", {"maneuver_type": "INTERCEPT"}),
    ("set_heading", {"direction": {"x": 1, "y": 0, "z": 0}}),
])
def test_maneuver_lasts_until_actual_checkpoint(fleet, interval, tool, args):
    sim, captain = setup_ships(fleet, interval=interval)
    maneuver = execute(captain, sim, tool, **args)
    assert maneuver.duration == interval
    assert not maneuver.is_complete(interval - 0.01)
    assert maneuver.is_complete(interval)


def test_prompt_and_tools_disclose_actual_interval(fleet):
    sim, captain = setup_ships(fleet, interval=60)
    captain.client.decide_with_tools.return_value = []
    captain.decide("a", sim)
    messages, tools = captain.client.decide_with_tools.call_args.args
    prompt = "\n".join(m["content"] for m in messages)
    assert "LOCKED for 60 seconds" in prompt
    assert "LOCKED for 30 seconds" not in prompt
    maneuver = next(t for t in tools if t["function"]["name"] == "set_maneuver")
    assert "30 seconds" not in maneuver["function"]["description"]


def test_guns_follow_primary_target_changes(fleet):
    sim, captain = setup_ships(fleet)
    execute(captain, sim, "set_primary_target", target_name="b")
    execute(captain, sim, "set_weapons_order", spinal_mode="FIRE_IMMEDIATE", turret_mode="FIRE_IMMEDIATE")
    execute(captain, sim, "set_primary_target", target_name="c")
    sim._process_weapons_orders(sim.get_ship("a"))
    assert sim.projectiles
    assert {p.target_ship_id for p in sim.projectiles} == {"c"}


@pytest.mark.parametrize("ship_type,count", [("corvette", 3), ("cruiser_torpedo", 12)])
def test_salvo_launches_requested_rounds_across_reload(fleet, ship_type, count):
    sim, captain = setup_ships(fleet, ship_type)
    # Keep the target out of terminal combat during the launch window.
    sim.get_ship("b").kinematic_state.position = Vector3D(2_000_000, 0, 0)
    assert captain._max_torpedo_salvo("a", sim) == count
    execute(captain, sim, "launch_torpedo", count=count, target_id="b")
    for _ in range(30):
        sim.step()
    launches = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_LAUNCHED]
    assert len(launches) == count
    assert sorted({e.timestamp for e in launches}) == [0, 12, 24]


def test_salvo_capacity_counts_each_tubes_magazine_and_reload(fleet):
    sim, captain = setup_ships(fleet, "cruiser_torpedo")
    tubes = sim.get_ship("a").ready_torpedo_launchers
    for tube, rounds in zip(tubes, [0, 1, 2, 12]):
        tube.current_magazine = rounds
        tube.last_launch_time = 0  # Next opportunities: 12 and 24, not 0.
    assert captain._max_torpedo_salvo("a", sim) == 5


def test_unreachable_accelerating_lead_does_not_spend_ammo(fleet):
    sim, _ = setup_ships(fleet)
    shooter = sim.get_ship("a")
    # Both trigger paths must handle a failed launch transactionally.
    slot = next(iter(shooter.weapons))
    weapon = shooter.weapons[slot]
    target = sim.get_ship("b")
    target.kinematic_state.position = Vector3D(6_000_000, 0, 0)
    target.current_maneuver = Maneuver(ManeuverType.BURN, 0, direction=Vector3D(0, 1, 0))
    before_ammo = weapon.ammo_remaining
    before_heat = shooter.heat_percent
    capacitor = shooter.power_system.weapon_capacitors[slot]
    before_charge = capacitor.current_charge_mj
    assert not sim.inject_command("a", {"type": "fire_at", "target_id": "b", "weapon_slot": slot})
    assert weapon.ammo_remaining == before_ammo
    assert weapon.cooldown_remaining == 0
    assert shooter.heat_percent == before_heat
    assert capacitor.current_charge_mj == before_charge
    shooter.weapons_orders[slot] = WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, slot, "b")
    sim._process_weapons_orders(shooter)
    assert weapon.ammo_remaining == before_ammo
    assert weapon.cooldown_remaining == 0
    assert capacitor.current_charge_mj == before_charge
    assert not sim.projectiles


def test_impossible_crossing_intercept_is_not_a_firing_solution(fleet):
    sim, _ = setup_ships(fleet)
    target = sim.get_ship("b")
    solution = calculate_hit_probability(
        Vector3D.zero(), Vector3D.zero(), target.position, Vector3D(0, 20_000, 0),
        target.geometry, target.forward, muzzle_velocity_kps=10,
    )
    assert not solution.can_fire
    assert solution.hit_probability == 0


def test_lead_intercepts_fast_crossing_target():
    # At 99% of muzzle speed, fixed-point iteration has not converged after ten iterations.
    lead = LeadCalculator.calculate_lead(Vector3D.zero(), Vector3D.zero(),
                                         Vector3D(100, 0, 0), Vector3D(0, 9.9, 0), 10)
    tof = lead.magnitude / 10
    assert (lead - Vector3D(100, 9.9 * tof, 0)).magnitude < 0.001


def test_projectile_collision_is_invariant_under_common_velocity(fleet):
    def outcome(boost):
        sim, _ = setup_ships(fleet)
        target = sim.get_ship("b")
        # The round is already ahead and pulling away. A world-space sweep
        # through the target's final position must not turn that miss into a hit.
        target.kinematic_state.velocity = Vector3D(boost, 0, 0)
        target.kinematic_state.position = Vector3D(0, 0, 0)
        sim.projectiles = [ProjectileInFlight(
            projectile_id="round", source_ship_id="a", target_ship_id="b",
            projectile=KineticProjectile(position=Vector3D(2_000, 0, 0),
                                          velocity=Vector3D(boost + 1_000, 0, 0),
                                          mass_kg=1, muzzle_velocity_kps=1),
        )]
        sim.step()
        return [e.event_type for e in sim.events if e.event_type in (
            SimulationEventType.PROJECTILE_IMPACT, SimulationEventType.PROJECTILE_MISS)]
    assert outcome(0) == outcome(10_000) == [SimulationEventType.PROJECTILE_MISS]


@pytest.mark.parametrize("overrides", [False, True])
def test_cli_preserves_fleet_options_and_applies_explicit_overrides(tmp_path, monkeypatch, overrides):
    from scripts import run_llm_battle as cli

    path = tmp_path / "fleet.json"
    path.write_text(json.dumps(fleet_dict(
        unlimited_mode=True, record_battle=False, record_sim_trace=True,
        personality_selection=False, decision_interval_s=45, initial_distance_km=750,
        time_limit_s=321,
    )))
    argv = ["run_llm_battle.py", "--fleet-config", str(path), "--quiet", "--seed", "123"]
    if overrides:
        argv += ["--distance", "600", "--time-limit", "90", "--decision-interval", "20"]
    monkeypatch.setattr(sys, "argv", argv)
    runner = Mock()
    runner.run_battle.return_value = SimpleNamespace(winner=None, reason="test", duration_s=0,
                                                    checkpoints_used=0, messages=[])
    factory = Mock(return_value=runner)
    monkeypatch.setattr(cli, "LLMBattleRunner", factory)
    client = Mock(side_effect=AssertionError("heuristic battle must not need an API key"))
    monkeypatch.setattr(cli, "CaptainClient", client)
    assert cli.main() == 0
    config = factory.call_args.kwargs["config"]
    assert config.unlimited_mode and config.record_sim_trace
    assert not config.record_battle and not config.personality_selection
    assert config.seed == 123
    assert config.initial_distance_km == (600 if overrides else 750)
    assert config.time_limit_s == (90 if overrides else 321)
    assert config.decision_interval_s == (20 if overrides else 45)
    assert factory.call_args.kwargs["fleet_config"].decision_interval_s == config.decision_interval_s


def test_cli_honors_captain_and_ship_names(monkeypatch):
    from scripts import run_llm_battle as cli

    monkeypatch.setattr(sys, "argv", ["run_llm_battle.py", "--alpha-model", "heuristic",
        "--beta-model", "heuristic", "--alpha-name", "Ada", "--alpha-ship", "Wasp",
        "--beta-name", "Grace", "--beta-ship", "Hornet"])
    factory = Mock()
    monkeypatch.setattr(cli, "LLMBattleRunner", factory)
    assert cli.main() == 0
    assert factory.call_args.kwargs["alpha_config"].name == "Ada"
    assert factory.call_args.kwargs["alpha_config"].ship_name == "Wasp"
    assert factory.call_args.kwargs["beta_config"].name == "Grace"
    assert factory.call_args.kwargs["beta_config"].ship_name == "Hornet"


@pytest.mark.parametrize("fleet_mode", [False, True])
def test_fractional_checkpoint_and_time_limit_are_exact(fleet, fleet_mode):
    config = BattleConfig(verbose=False, record_battle=False, personality_selection=False,
                          decision_interval_s=20.5, time_limit_s=61.25)
    fc = BattleFleetConfig.from_dict(fleet_dict(decision_interval_s=20.5, time_limit_s=61.25)) if fleet_mode else None
    runner = LLMBattleRunner(config, LLMCaptainConfig("A", "A", model="heuristic"),
                            LLMCaptainConfig("B", "B", model="heuristic"), None, fc)
    result = runner.run_battle(fleet)
    assert result.duration_s == 61.25
    assert [decision["time"] for decision in result.decision_log] == [20.5, 41]
    assert runner.simulation.time_step == 1


@pytest.mark.parametrize("stop", ["destroyed", "dying", "expired"])
def test_queued_salvos_cancel_when_no_longer_valid(fleet, stop):
    sim, captain = setup_ships(fleet, "corvette")
    execute(captain, sim, "launch_torpedo", count=3, target_id="b")
    assert len(sim.pending_torpedo_launches) == 2
    if stop == "expired":
        sim.current_time = 30
    else:
        setattr(sim.get_ship("b"), f"is_{stop}", True)
        sim.current_time = 12
    sim._process_pending_torpedo_launches()
    assert not sim.pending_torpedo_launches
    assert sim.metrics.total_torpedoes_launched == 1


def test_hull_specific_torpedo_tool_capacity_does_not_mutate_shared_schema(fleet):
    from src.llm.tools import get_captain_tools_for_ship

    def capacity(ship_type, interval):
        tools = get_captain_tools_for_ship(ship_type, fleet, True, interval)
        return next(t for t in tools if t["function"]["name"] == "launch_torpedo")["function"]["parameters"]["properties"]["count"]["maximum"]

    assert capacity("corvette", 30) == 3
    assert capacity("cruiser_torpedo", 60) == 20
    assert capacity("corvette", 30) == 3


@pytest.mark.parametrize("velocity,accel,expected", [
    (Vector3D(-20, 0, 0), Vector3D.zero(), 100 / 30),
    (Vector3D(-10, 0, 0), Vector3D.zero(), 5),
    (Vector3D(10, 0, 0), Vector3D.zero(), None),
    (Vector3D(0, 11, 0), Vector3D.zero(), None),
    (Vector3D.zero(), Vector3D(0.1, 0, 0), (10 - math.sqrt(80)) / 0.1),
    (Vector3D.zero(), Vector3D(2, 0, 0), None),
])
def test_intercept_solver_selects_first_future_root(velocity, accel, expected):
    result = intercept_time(Vector3D(100, 0, 0), velocity, 10, accel)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, rel=1e-9)
        aim = Vector3D(100, 0, 0) + velocity * result + accel * (0.5 * result ** 2)
        assert aim.magnitude == pytest.approx(10 * result)


def test_accelerating_high_closure_lead_remains_finite_and_hits(fleet):
    sim, _ = setup_ships(fleet)
    gun = next(iter(sim.get_ship("a").weapons.values()))
    velocity = Vector3D(-30_000, 2_000, 0)
    accel = Vector3D(0, 20, 0)
    position = Vector3D(100_000, 0, 0)
    direction = gun.calculate_fire_direction(Vector3D(1, 0, 0), Vector3D.zero(), position,
                                             velocity, Vector3D.zero(), accel)
    assert direction is not None
    speed = gun.weapon.muzzle_velocity_kps * 1000
    tof = intercept_time(position, velocity, speed, accel)
    target_at_impact = position + velocity * tof + accel * (0.5 * tof ** 2)
    assert (direction * speed * tof - target_at_impact).magnitude < 0.001


def test_fuel_depletion_limits_rotational_impulse():
    state = ShipState(mass_kg=5_000.5, dry_mass_kg=5_000, propellant_kg=0.5,
                      thrust_n=1_000_000, exhaust_velocity_ms=1_000_000,
                      moment_of_inertia_kg_m2=1_000_000)
    # Only 0.5 seconds of fuel. A ten-second integration step must not buy
    # twenty times more thrust-vectoring impulse than a half-second step.
    short = propagate_state(state, 0.5, throttle=1, gimbal_pitch_deg=1)
    long = propagate_state(state, 10, throttle=1, gimbal_pitch_deg=1)
    assert long.angular_velocity.y == pytest.approx(short.angular_velocity.y)


def test_torque_uses_the_same_gimbal_limits_as_linear_thrust():
    state = ShipState()
    clamped = calculate_torque_from_thrust(state, gimbal_pitch_deg=MAX_GIMBAL_ANGLE_DEG)
    over = calculate_torque_from_thrust(state, gimbal_pitch_deg=90)
    assert over.y == pytest.approx(clamped.y)


@pytest.mark.parametrize("ship_type", ["corvette", "cruiser_torpedo", "destroyer"])
def test_intact_torpedo_ships_are_combat_effective(fleet, ship_type):
    from src.llm.victory import VictoryEvaluator
    sim, _ = setup_ships(fleet, ship_type)
    assert not VictoryEvaluator().is_ship_disabled(sim.get_ship("a"))


def test_simultaneous_surrender_is_a_draw(fleet):
    from src.llm.victory import VictoryEvaluator, BattleOutcome
    sim, _ = setup_ships(fleet)
    outcome, winner, _ = VictoryEvaluator().evaluate(sim.get_ship("a"), sim.get_ship("b"),
                                                     alpha_surrendered=True, beta_surrendered=True)
    assert outcome == BattleOutcome.DRAW
    assert winner is None


def test_mcp_maneuver_uses_actual_decision_window(fleet):
    from src.llm.mcp_controller import apply_mcp_commands_to_simulation
    from src.llm.mcp_state import MCPCommand, MCPCommandType
    sim, _ = setup_ships(fleet, interval=60)
    result = apply_mcp_commands_to_simulation([
        MCPCommand(MCPCommandType.SET_MANEUVER, "a", {"maneuver_type": "INTERCEPT", "target_id": "b"}),
    ], sim, "alpha")
    assert not result["errors"]
    assert sim.get_ship("a").current_maneuver.duration == 60
