"""
Regression tests for the pass-5 damage-lane review findings
(src/battle_damage.py and its hooks). One section per finding.
"""

import json
from pathlib import Path

import pytest

from src import battle_damage as bd
from src.combat import HitLocation
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    SimulationEventType,
    create_ship_from_fleet_data,
)

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "fleet_ships.json") as f:
    FLEET = json.load(f)


def _ship(sim, ship_id, faction, position, ship_type="cruiser",
          velocity=None, forward=None):
    ship = create_ship_from_fleet_data(
        ship_id, ship_type, faction, FLEET, position=position,
        velocity=velocity or Vector3D(0, 0, 0), forward=forward or Vector3D(1, 0, 0))
    sim.add_ship(ship)
    return ship


def _events(sim, event_type):
    return [e for e in sim.events if e.event_type == event_type]


def _kill_drive(ship):
    for module in ship._get_modules_by_type("engine"):
        module.health_percent = 0.0


# ---------------------------------------------------------------------------
# Finding 1: an adrift hull must not hold back the formation's burn
# ---------------------------------------------------------------------------

def _formation_pair(adrift):
    sim = CombatSimulation(seed=1, time_step=1.0)
    lead = _ship(sim, "a1", "alpha", Vector3D(0, 0, 0))
    # the wingman faces away: a 90 deg turn to the ordered heading (+y)
    wing = _ship(sim, "a2", "alpha", Vector3D(0, 10_000, 0), forward=Vector3D(-1, 0, 0))
    _ship(sim, "e1", "beta", Vector3D(9e8, 0, 0))
    if adrift:
        _kill_drive(wing)
    return sim, lead, wing


def test_adrift_hull_does_not_delay_the_formation_burn_start():
    from src.llm.range_control import formation_heading_maneuvers

    sim, lead, wing = _formation_pair(adrift=False)
    intact = formation_heading_maneuvers(sim, [lead, wing], {"x": 0, "y": 1, "z": 0}, 1.0)
    sim_a, lead_a, wing_a = _formation_pair(adrift=True)
    alone = formation_heading_maneuvers(sim_a, [lead_a], {"x": 0, "y": 1, "z": 0}, 1.0)
    orders = formation_heading_maneuvers(sim_a, [lead_a, wing_a], {"x": 0, "y": 1, "z": 0}, 1.0)

    assert orders["a2"].throttle == 0.0
    # the burn lights when the hulls that CAN burn are nose-on - not after
    # the adrift wingman's ~2-minute RCS turn
    assert orders["a1"].burn_start_time == pytest.approx(alone["a1"].burn_start_time)
    assert orders["a1"].burn_start_time - sim_a.current_time < 30.0
    assert intact["a1"].burn_start_time - sim.current_time < 30.0

    # and the engine flies it: the lead gains real speed in one checkpoint
    v0 = lead_a.velocity
    for sid, maneuver in orders.items():
        sim_a.get_ship(sid).current_maneuver = maneuver
    sim_a._running = True
    sim_a.current_time = 0.001          # past the t=0 decision trigger
    for _ in range(30):
        sim_a.step()
    assert (lead_a.velocity - v0).magnitude > 100.0


def test_adrift_hull_does_not_inflate_range_control_burn_times():
    from src.llm import range_control as rc

    sim, lead, wing = _formation_pair(adrift=True)
    h = Vector3D(1, 0, 0)       # the lead is nose-on, the wingman faces away
    assert rc._burn_start(sim, [lead, wing], h) == pytest.approx(rc._burn_start(sim, [lead], h))
    assert rc._turn_time(sim, [lead, wing], h, engines_on=True) == pytest.approx(
        rc._turn_time(sim, [lead], h, engines_on=True))
    # the attitude-only (RCS) turn still counts the adrift hull
    assert rc._turn_time(sim, [lead, wing], h, engines_on=False) > rc._turn_time(
        sim, [lead], h, engines_on=False)
    # a lone adrift hull: nothing to wait for
    assert rc._burn_start(sim, [wing], h) == 0.0


# ---------------------------------------------------------------------------
# Finding 2: the blast goes off where the dead ship is after this tick's move
# ---------------------------------------------------------------------------

def _co_moving_pair():
    sim = CombatSimulation(seed=1, time_step=1.0)
    v = Vector3D(5_000, 0, 0)
    dead = _ship(sim, "hulk", "alpha", Vector3D(0, 0, 0), velocity=v)
    near = _ship(sim, "near", "alpha", Vector3D(0, 5_000, 0), velocity=v)
    _ship(sim, "enemy", "beta", Vector3D(9e8, 0, 0))
    sim._running = True
    sim.current_time = 0.001            # past the t=0 decision trigger
    return sim, dead, near


def test_fuse_expiry_blast_origin_moves_with_the_hulk():
    sim, hulk, near = _co_moving_pair()
    sim._start_death_spiral(hulk, "enemy")
    hulk.dying_time_remaining = 0.5
    hulk.dying_torch_on = False
    sim.step()
    blast = _events(sim, SimulationEventType.BLAST_DAMAGE)
    assert hulk.is_destroyed and len(blast) == 1
    assert blast[0].data["distance_km"] == pytest.approx(5.0, abs=0.01)
    origin = _events(sim, SimulationEventType.REACTOR_BLAST)[0].data["position"]
    assert origin[0] == pytest.approx(near.position.x, abs=1.0)


def test_meltdown_blast_origin_moves_with_the_ship():
    sim, hot, near = _co_moving_pair()
    hot.thermal_system.heatsink.current_heat_gj = 1.6 * hot.thermal_system.heatsink.capacity_gj
    sim.step()
    assert hot.is_destroyed
    blast = _events(sim, SimulationEventType.BLAST_DAMAGE)
    assert len(blast) == 1
    assert blast[0].data["distance_km"] == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# Finding 3: a blast's heat pulse alone never melts a reactor down
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heat0", [0.0, 1.0, 1.2, 1.45, 1.495])
def test_blast_heat_stops_just_short_of_the_meltdown_line(heat0):
    from src.thermal import REACTOR_MELTDOWN_HEAT_FRACTION

    sim = CombatSimulation(seed=1, time_step=1.0)
    dead = [_ship(sim, f"dead{k}", "alpha", Vector3D(0, -1_500 * k, 0)) for k in range(2)]
    victim = _ship(sim, "victim", "alpha", Vector3D(0, 1_000, 0))
    _ship(sim, "enemy", "beta", Vector3D(9e8, 0, 0))
    sink = victim.thermal_system.heatsink
    sink.current_heat_gj = heat0 * sink.capacity_gj
    sim._running = True
    sim.current_time = 0.001
    for ship in dead:                   # two close blasts in the same tick
        sim._destroy_ship(ship, "enemy", cause="reactor_destroyed")
    bd.resolve_pending_blasts(sim)

    assert victim.thermal_system.heat_fraction < REACTOR_MELTDOWN_HEAT_FRACTION
    events = _events(sim, SimulationEventType.BLAST_DAMAGE)
    assert len(events) == 2
    if heat0 == 0.0:                     # the uncapped pulse when there is room
        assert events[0].data["heat_gj"] == pytest.approx(
            events[0].data["intensity"] * bd.REACTOR_BLAST_MAX_HEAT_GJ, abs=0.1)
    assert all(e.data["heat_gj"] >= 0.0 for e in events)
    # the next tick: no meltdown, the ship lives on
    sim.step()
    assert not victim.is_destroyed
    assert not _events(sim, SimulationEventType.REACTOR_MELTDOWN)


# ---------------------------------------------------------------------------
# Finding 5: recorded health 0.0 always means destroyed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["gun", "pd", "launcher", "engine"])
@pytest.mark.parametrize("loss", [99.4, 99.5, 99.96, 100.0])
def test_a_fitting_left_under_half_a_percent_is_destroyed(kind, loss):
    sim = CombatSimulation(seed=1, time_step=1.0)
    hull = "cruiser_torpedo" if kind == "launcher" else "cruiser"
    ship = _ship(sim, "victim", "beta", Vector3D(0, 0, 0), ship_type=hull)
    if kind == "gun":
        fitting = bd._gun_fittings(ship, HitLocation.LATERAL)[0]
    elif kind == "pd":
        fitting = bd._pd_fittings(ship)[0]
    elif kind == "launcher":
        fitting = bd._launcher_fittings(ship)[0]
    else:
        module = ship._get_modules_by_type("engine")[0]
        fitting = ("engine", module.name, module)
    bd.damage_fitting(sim, ship, fitting, loss, "surface_hit", "x", HitLocation.LATERAL)
    data = _events(sim, SimulationEventType.MOUNT_DAMAGED)[-1].data
    assert data["destroyed"] == (data["health_pct"] == 0.0)
    assert data["destroyed"] == (loss > 100.0 - bd.FITTING_DESTROYED_BELOW_PCT)
    state = fitting[2]
    health = state.health_percent if kind == "engine" else state.mount_health
    assert (health == 0.0) == data["destroyed"]
    if kind != "engine":
        assert state.is_operational is (not data["destroyed"])


# ---------------------------------------------------------------------------
# Finding 4: a blast-blinded torpedo that hits is not a PD seeker kill
# ---------------------------------------------------------------------------

def _torpedo_flight(sim, position, velocity, target_id, torpedo_id="t1"):
    from src.simulation import TorpedoInFlight
    from src.torpedo import Torpedo, TorpedoSpecs

    torp = Torpedo(specs=TorpedoSpecs(), position=position, velocity=velocity,
                   target_id=target_id)
    torp.armed = True
    flight = TorpedoInFlight(torpedo_id=torpedo_id, torpedo=torp,
                             source_ship_id="shooter", launch_time=sim.current_time)
    sim.torpedoes.append(flight)
    return flight


def test_blast_blinded_torpedo_hit_stays_out_of_the_pd_tally():
    sim = CombatSimulation(seed=1, time_step=1.0)
    _ship(sim, "shooter", "alpha", Vector3D(500_000, 0, 0))
    target = _ship(sim, "victim", "beta", Vector3D(0, 0, 0), ship_type="battleship")
    dead = _ship(sim, "dead", "beta", Vector3D(0, -5_000, 0))
    # 6 km from the cruiser's blast: inside the blind ring, outside the kill ring
    flight = _torpedo_flight(sim, Vector3D(0, 1_000, 0), Vector3D(0, 5_000, 0), "victim")
    sim._destroy_ship(dead, "shooter", cause="reactor_destroyed")
    bd.resolve_pending_blasts(sim)
    assert flight.is_disabled and flight.blast_blinded
    assert sim.metrics.total_torpedo_blast_blinded == 1

    sim._resolve_torpedo_hit(flight, target)
    m = sim.metrics
    assert m.total_torpedo_hits == 1
    assert m.total_torpedo_hits_after_blast_blind == 1
    assert m.total_torpedo_seeker_killed == 0
    assert m.total_torpedo_hits_after_seeker_kill == 0   # PD's tally is untouched
    assert "(0 of those still hit)" in m.torpedo_outcome_summary()


def test_pd_blinded_torpedo_hit_still_counts_as_a_seeker_kill_that_hit():
    sim = CombatSimulation(seed=1, time_step=1.0)
    _ship(sim, "shooter", "alpha", Vector3D(500_000, 0, 0))
    target = _ship(sim, "victim", "beta", Vector3D(0, 0, 0), ship_type="battleship")
    flight = _torpedo_flight(sim, Vector3D(0, -1_000, 0), Vector3D(0, 5_000, 0), "victim")
    flight.is_disabled = True            # PD's work
    sim._resolve_torpedo_hit(flight, target)
    assert sim.metrics.total_torpedo_hits_after_seeker_kill == 1
    assert sim.metrics.total_torpedo_hits_after_blast_blind == 0


# ---------------------------------------------------------------------------
# Finding 9: resolving blasts never queues another one in the same tick
# (why step() needs no last-tick flush: chain deaths happen on a later tick)
# ---------------------------------------------------------------------------

def test_blast_resolution_never_queues_a_blast_and_the_last_tick_needs_no_flush():
    import inspect

    sim = CombatSimulation(seed=1, time_step=1.0)
    dead = _ship(sim, "dead", "alpha", Vector3D(0, 0, 0))
    hulk = _ship(sim, "hulk", "alpha", Vector3D(0, 500, 0))
    hot = _ship(sim, "hot", "beta", Vector3D(0, -500, 0))
    sim._start_death_spiral(hulk, "x")
    hulk.dying_time_remaining = 1.0                  # the blast burns it to 0
    hot.thermal_system.heatsink.current_heat_gj = 1.49 * hot.thermal_system.heatsink.capacity_gj
    sim._destroy_ship(dead, "x", cause="reactor_destroyed")
    bd.resolve_pending_blasts(sim)
    assert hulk.dying_time_remaining == 0.0 and not hulk.is_destroyed
    assert sim._pending_blasts == []                 # nothing new queued
    assert "while self._pending_blasts" not in inspect.getsource(CombatSimulation.step)


# ---------------------------------------------------------------------------
# Finding 6: commanders see degraded (not yet wrecked) fittings
# ---------------------------------------------------------------------------

def _degraded_fleet():
    sim = CombatSimulation(time_step=1.0, decision_interval=30, seed=5)
    cruiser = _ship(sim, "alpha_1", "alpha", Vector3D(0, 0, 0))
    tubes = _ship(sim, "alpha_2", "alpha", Vector3D(0, 8_000, 0), ship_type="cruiser_torpedo")
    _ship(sim, "beta_1", "beta", Vector3D(300_000, 0, 0), ship_type="destroyer",
          forward=Vector3D(-1, 0, 0))
    for ship in (cruiser, tubes):
        ship.name = f"TIS {ship.ship_type}-{ship.ship_id[-1]}"
    gun = ("gun", "weapon_1", cruiser.weapons["weapon_1"])
    bd.damage_fitting(sim, cruiser, gun, 54.0, "surface_hit", "beta_1", HitLocation.LATERAL)
    bd.damage_fitting(sim, cruiser, bd._pd_fittings(cruiser)[0], 30.0, "surface_hit",
                      "beta_1", HitLocation.LATERAL)
    bd.damage_fitting(sim, tubes, bd._launcher_fittings(tubes)[2], 60.0, "surface_hit",
                      "beta_1", HitLocation.LATERAL)
    return sim, cruiser, tubes


def test_damaged_fitting_lines_name_health_and_cycle():
    sim, cruiser, tubes = _degraded_fleet()
    lines = bd.damaged_fitting_lines(cruiser)
    assert lines == ["Coilgun Battery Mk3 [weapon_1] DAMAGED 46 %, cycle x2.0",
                     f"PD Laser [{cruiser.point_defense[0].turret_name}] DAMAGED 70 %, full output"]
    assert bd.damaged_fitting_lines(tubes) == [
        "Torpedo Launcher [weapon_2] DAMAGED 40 %, cycle x2.0"]
    # wrecked fittings are not 'damaged': they already show as destroyed/LOST
    bd.damage_fitting(sim, cruiser, ("gun", "weapon_1", cruiser.weapons["weapon_1"]),
                      100.0, "surface_hit", "beta_1", HitLocation.LATERAL)
    assert not any("weapon_1" in line for line in bd.damaged_fitting_lines(cruiser))


def test_captain_admiral_and_mcp_views_show_degraded_fittings():
    import types

    from src.llm.admiral import LLMAdmiral
    from src.llm.captain import LLMCaptain, LLMCaptainConfig
    from src.llm.mcp_controller import MCPController, MCPControllerConfig
    from src.llm.prompts import format_admiral_turn_state, format_my_ship

    sim, cruiser, tubes = _degraded_fleet()
    gun_line = "Coilgun Battery Mk3 [weapon_1] DAMAGED 46 %, cycle x2.0"
    tube_line = "Torpedo Launcher [weapon_2] DAMAGED 40 %, cycle x2.0"

    # captain: status dict -> the weapons line of its own-ship block
    config = LLMCaptainConfig(name="Captain", ship_name=cruiser.name, ship_type="cruiser",
                              fleet_data=FLEET, model="heuristic")
    captain = LLMCaptain(config, client=object())
    captain.ship_id = cruiser.ship_id
    status = captain._build_ship_status(cruiser)
    assert gun_line in status["damaged_fittings"]
    text = format_my_ship("Captain", cruiser.name, "cruiser", status, {}, FLEET)
    assert gun_line in text

    # admiral: fleet table
    captains = [types.SimpleNamespace(
        ship_id=s.ship_id, ship_name=s.name, name=f"Captain {s.ship_id}",
        config=types.SimpleNamespace(ship_type=s.ship_type, name=f"Captain {s.ship_id}"))
        for s in (cruiser, tubes)]
    admiral = LLMAdmiral(types.SimpleNamespace(
        name="Admiral alpha", model="x/y", personality=None, vision=False,
        notebook_text=None, image_input=False), "alpha", client=None, fleet_data=FLEET)
    snapshot = admiral._build_snapshot(sim, captains, full=True)
    fleet_text = format_admiral_turn_state(snapshot, FLEET, phase="directive")
    assert gun_line in fleet_text and tube_line in fleet_text

    # MCP: per-weapon health and cycle factor, and the same lines
    state = MCPController(MCPControllerConfig(faction="alpha"), FLEET).build_state_for_mcp(
        sim, captains)
    blob = json.dumps(state.friendly_ships, default=str)
    assert gun_line in blob and tube_line in blob
    assert '"mount_health_pct": 46.0' in blob and '"cycle_factor": 2.0' in blob


# ---------------------------------------------------------------------------
# Finding 7: combat-ineffective fleets
# ---------------------------------------------------------------------------

def _fleet_runner(alpha_types, beta_types, **options):
    from unittest.mock import Mock

    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    from src.llm.fleet_config import BattleFleetConfig

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
    runner.setup_fleet_battle(load_fleet_data("data/fleet_ships.json"))
    return runner


def _disarm(sim, ship):
    """Every gun mount and torpedo tube wrecked (PD and drive untouched)."""
    for fitting in bd._gun_fittings(ship, None) + bd._launcher_fittings(ship):
        bd.damage_fitting(sim, ship, fitting, 100.0, "surface_hit", "x", HitLocation.LATERAL)


def _set_adrift(sim, ship):
    for module in ship._get_modules_by_type("engine"):
        bd.damage_fitting(sim, ship, ("engine", module.name, module), 100.0,
                          "surface_hit", "x", HitLocation.TAIL)


def test_disarmed_and_adrift_survivors_score_less_at_the_time_limit():
    from src.llm.battle_runner import (ADRIFT_SHIP_SCORE_FACTOR,
                                       DISARMED_SHIP_SCORE_FACTOR,
                                       surviving_ship_score_factor)

    runner = _fleet_runner(["cruiser", "cruiser"], ["cruiser", "cruiser"])
    sim = runner.simulation
    a1, a2 = runner.alpha_ships["alpha_1"], runner.alpha_ships["alpha_2"]
    assert surviving_ship_score_factor(a1) == 1.0
    outcome, _, reason = runner._evaluate_fleet_tactical_advantage()
    assert outcome.value == "draw", reason          # mirror fleets, untouched
    _disarm(sim, a1)
    _set_adrift(sim, a2)
    assert a1.hull_integrity == 100.0               # the hulls look intact...
    assert surviving_ship_score_factor(a1) == DISARMED_SHIP_SCORE_FACTOR
    assert surviving_ship_score_factor(a2) == ADRIFT_SHIP_SCORE_FACTOR
    outcome, winner, reason = runner._evaluate_fleet_tactical_advantage()
    assert winner == "beta", reason                 # ...but alpha can barely fight


def test_fleet_battle_ends_when_one_side_can_no_longer_deal_damage():
    runner = _fleet_runner(["corvette", "frigate"], ["destroyer"])
    sim = runner.simulation
    assert not runner._is_fleet_battle_over()       # latches: both armed at start
    for ship in runner.alpha_ships.values():
        _disarm(sim, ship)
    assert runner._is_fleet_battle_over()
    result = runner._evaluate_fleet_result()
    assert result.winner == "beta"
    assert "Alpha fleet combat-ineffective" in result.reason


def test_a_torpedo_still_in_flight_keeps_its_fleet_in_the_fight():
    runner = _fleet_runner(["corvette"], ["destroyer"])
    sim = runner.simulation
    assert not runner._is_fleet_battle_over()
    corvette = runner.alpha_ships["alpha_1"]
    flight = _torpedo_flight(sim, Vector3D(1e5, 0, 0), Vector3D(0, 0, 0), "beta_1")
    flight.source_ship_id = corvette.ship_id
    _disarm(sim, corvette)
    assert not runner._is_fleet_battle_over()
    sim.torpedoes.remove(flight)
    assert runner._is_fleet_battle_over()


def test_both_fleets_combat_ineffective_ends_on_points():
    runner = _fleet_runner(["frigate"], ["frigate"])
    sim = runner.simulation
    assert not runner._is_fleet_battle_over()
    for ship in list(runner.alpha_ships.values()) + list(runner.beta_ships.values()):
        _disarm(sim, ship)
    assert runner._is_fleet_battle_over()
    result = runner._evaluate_fleet_result()
    assert result.reason.startswith("Both fleets combat-ineffective - on points")
    assert result.outcome.value == "draw"            # mirror hulls, same points


def test_a_fleet_unarmed_from_the_start_does_not_end_the_battle():
    runner = _fleet_runner(["frigate"], ["frigate"])
    sim = runner.simulation
    for ship in runner.alpha_ships.values():        # unarmed before the first look
        for ws in ship.weapons.values():
            ws.ammo_remaining = 0
    assert not runner._is_fleet_battle_over()
    assert runner._fleet_armed_at_start == {"alpha": False, "beta": True}
    sim.step()
    assert not runner._is_fleet_battle_over()


def test_one_v_one_victory_rules_are_unchanged():
    from src.llm.victory import VictoryEvaluator
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _ship(sim, "alpha", "alpha", Vector3D(0, 0, 0), ship_type="frigate")
    assert VictoryEvaluator().is_ship_disabled(ship) is False
    _disarm(sim, ship)
    for pd in ship.point_defense:
        pd.is_operational = False
    for slot, ws in ship.weapons.items():
        if slot.startswith("pd"):
            ws.is_operational = False
    assert VictoryEvaluator().is_ship_disabled(ship) is True
