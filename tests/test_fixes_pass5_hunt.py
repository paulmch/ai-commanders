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
    # Pass 5 (balance): a hull the draft prices is valued at that price, so a
    # variant is no longer worth exactly its base hull; the table is the
    # fallback for hulls the price table does not know.
    from src.llm.fleet_draft import SHIP_POINT_COSTS
    for ship_type in hulls:
        assert ship_class_value(ship_type) == SHIP_POINT_COSTS[ship_type]
    assert ship_class_value("cruiser_refit") == SHIP_CLASS_VALUES["cruiser"]
    assert ship_class_value("dreadnought_x") == SHIP_CLASS_VALUES["dreadnought"]
    assert ship_class_value("monitor") == SHIP_CLASS_VALUES["destroyer"]


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


# ---------------------------------------------------------------------------
# 4. A replaced weapons order is 'superseded' once, not once per gun.
#
# inject_command binds a weapons order to one evidence key per weapon slot and
# logged the replaced command 'superseded' for every key: 18-83 duplicate
# command_status events per seeded battle (a cruiser's three batteries logged
# the same command superseded three times in the same tick).
# ---------------------------------------------------------------------------

def test_replaced_weapons_order_is_superseded_once():
    from src.firecontrol import WeaponsCommand, WeaponsOrder
    from src.simulation import SimulationEventType
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    sim = _torpedo_sim(data)
    ship = sim.get_ship("atk")
    slots = [s for s, w in ship.weapons.items() if not s.startswith("pd")]
    assert len(slots) >= 2

    def order():
        return {"type": "weapons_orders", "default_target_id": "tgt",
                "orders": [WeaponsOrder(command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
                                        weapon_slot=s, target_id="tgt")
                           for s in slots]}

    assert sim.inject_command("atk", order())
    assert sim.inject_command("atk", order())
    superseded = [e for e in sim.events
                  if e.event_type == SimulationEventType.COMMAND_STATUS
                  and e.data.get("status") == "superseded"]
    assert len(superseded) == 1



# ---------------------------------------------------------------------------
# 5. Gun hits and misses name the weapon that fired them.
#
# PROJECTILE_IMPACT / PROJECTILE_MISS carried no weapon_slot, so every
# recorded 'hit' and 'miss' said weapon_slot "unknown" (all 1,549 hits in 10
# seeded battles), every LLM captain's shot history read "You fired unknown:
# 230km ... -> HIT", and DAMAGE TAKEN named the enemy weapon "unknown". The
# slot recorded at launch now rides on the round's impact and miss events.
# ---------------------------------------------------------------------------

def test_gun_hit_and_miss_events_carry_the_firing_slot():
    from src.firecontrol import WeaponsCommand, WeaponsOrder
    from src.physics import Vector3D
    from src.simulation import CombatSimulation, SimulationEventType, create_ship_from_fleet_data
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=11)
    for sid, faction, x, fwd in (("a", "alpha", 0.0, 1.0), ("b", "beta", 60_000.0, -1.0)):
        sim.add_ship(create_ship_from_fleet_data(
            ship_id=sid, ship_type="destroyer", faction=faction, fleet_data=data,
            position=Vector3D(x, 0, 0), velocity=Vector3D(0, 0, 0),
            forward=Vector3D(fwd, 0, 0)))
    shooter = sim.get_ship("a")
    shooter.primary_target_id = "b"
    slots = [s for s in shooter.weapons if not s.startswith("pd")]
    sim.inject_command("a", {"type": "weapons_orders", "default_target_id": "b",
                             "orders": [WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE,
                                                     weapon_slot=s, target_id="b")
                                        for s in slots]})
    for _ in range(60):
        sim.step()
    launched = {e.data["projectile_id"]: e.data["weapon_slot"] for e in sim.events
                if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED}
    resolved = [e for e in sim.events
                if e.event_type in (SimulationEventType.PROJECTILE_IMPACT,
                                    SimulationEventType.PROJECTILE_MISS)
                and e.data.get("projectile_id") in launched]
    assert resolved, "no gun round resolved in 60 s at 60 km"
    for e in resolved:
        assert e.data.get("weapon_slot") == launched[e.data["projectile_id"]], e.data



# ---------------------------------------------------------------------------
# Confirmed, not fixed here (reported to the orchestrator).
# ---------------------------------------------------------------------------

def _battleship_sim():
    from src.physics import Vector3D
    from src.simulation import CombatSimulation, create_ship_from_fleet_data
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=1)
    ship = create_ship_from_fleet_data(
        ship_id="bb", ship_type="battleship", faction="alpha", fleet_data=data,
        position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0),
        forward=Vector3D(1, 0, 0))
    sim.add_ship(ship)
    return sim, ship


# Rotation used to keep only a scalar |omega| re-pointed at the target every
# tick, so a turn could reverse - or stop dead on the target - at any rate: a
# battleship spun up on thrust vectoring to 0.047 rad/s and then turning on
# RCS (7.5e-5 rad/s^2) toward a point 1 deg behind its nose turned -0.0175
# rad/s the next tick (~850x its authority). Seen in seed 5 budget 150:
# battleship alpha_1 in PADLOCK at t=281-282. The spin is now a vector that
# only the available authority can change.
def test_rcs_turn_cannot_reverse_a_thrust_vectored_spin_in_one_tick():
    import math
    from src.physics import Vector3D
    sim, ship = _battleship_sim()
    rcs_alpha = math.radians(ship.attitude_control.rcs_angular_accel_deg_s2)
    for _ in range(10):  # thrust-vectored turn toward +y builds a rate
        sim._rotate_ship_toward(ship, Vector3D(0, 1, 0), 1.0, engines_on=True)
    before = ship.forward
    rate_before = ship.angular_velocity_rad_s          # about +z
    assert rate_before > 0.03
    behind = before.rotate_around_axis(Vector3D(0, 0, 1), -math.radians(1.0))
    sim._rotate_ship_toward(ship, behind, 1.0, engines_on=False)
    signed_rate = math.asin(before.cross(ship.forward).z)   # rad over 1 s
    assert signed_rate >= rate_before - 2.0 * rcs_alpha


# A destroyed ship used to freeze in space (step() skipped it) while its
# recorded velocity stayed at the last value: seed 1 budget 150 beta_5
# detonated at t=132 moving 1.25 km/s and its trace never moved again. The
# wreck now coasts; nothing that fights may treat it as alive.
def test_destroyed_wreck_keeps_drifting():
    from src.physics import Vector3D
    sim, ship = _battleship_sim()
    ship.kinematic_state.velocity = Vector3D(1000.0, 0, 0)
    ship.is_destroyed = True
    x0 = ship.position.x
    sim.step()
    assert ship.position.x == pytest.approx(x0 + 1000.0, abs=1.0)
    sim.step()
    assert ship.position.x == pytest.approx(x0 + 2000.0, abs=1.0)


def test_hulk_keeps_moving_on_its_detonation_tick():
    from src.physics import Vector3D
    sim, ship = _battleship_sim()
    ship.kinematic_state.velocity = Vector3D(1000.0, 0, 0)
    sim._start_death_spiral(ship, None)
    ship.dying_time_remaining = 0.5
    ship.dying_torch_on = False
    x0 = ship.position.x
    sim.step()
    assert ship.is_destroyed
    assert ship.position.x == pytest.approx(x0 + 1000.0, abs=1.0)


def test_drifting_wreck_stays_out_of_the_fight():
    from src.physics import Vector3D
    from src.simulation import SimulationEventType
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    sim = _torpedo_sim(data)
    wreck = sim.get_ship("tgt")
    wreck.is_destroyed = True
    wreck.kinematic_state.velocity = Vector3D(-2000.0, 0, 0)
    # A ballistic round on a collision course with the drifting wreck.
    flight = _inject_torpedo(sim, Vector3D(-20_000.0, 0, 0), Vector3D(2000.0, 0, 0), disabled=True)
    for _ in range(8):
        sim.step()
    assert wreck.position.x < -10_000.0          # it moved through the round's path
    assert flight in sim.torpedoes                # ...and was not struck
    assert sim.get_enemy_ships("atk") == []
    assert not any(e.event_type in (SimulationEventType.TORPEDO_IMPACT,
                                    SimulationEventType.SHIP_DESTROYED)
                   for e in sim.events)


# ---------------------------------------------------------------------------
# 6. Auto-draft spends its budget when the hull cap binds.
#
# auto_draft ("greedily fills the budget") cycles its doctrine's hulls until
# the 8-hull cap, then stops: the gunline doctrine fielded the same 144-point
# fleet at budgets 150, 200, 250 and 300 - in the seeded 250-point battle
# (seed 10) alpha fought with 144 points against beta's 222. Once the cap
# binds, the cheapest hulls are now upgraded to the doctrine's dearer ones
# while the budget allows.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("budget", [200, 250, 300])
def test_capped_auto_draft_upgrades_instead_of_leaving_points_unspent(seed, budget):
    from src.llm.fleet_draft import SHIP_POINT_COSTS, auto_draft
    draft = auto_draft("alpha", budget=budget, max_ships=8, seed=seed)
    types = [s.ship_type for s in draft.ships]
    assert draft.points_spent == sum(SHIP_POINT_COSTS[t] for t in types) <= budget
    assert len(types) <= 8
    left = budget - draft.points_spent
    if len(types) == 8:
        # No hull could be swapped for a dearer one of the same doctrine.
        doctrine = set(types)
        dearest = max(SHIP_POINT_COSTS[t] for t in doctrine)
        assert all(SHIP_POINT_COSTS[t] == dearest
                   or dearest - SHIP_POINT_COSTS[t] > left for t in types), (types, left)


def test_uncapped_auto_draft_is_unchanged():
    from src.llm.fleet_draft import auto_draft
    # Budget 100 never reaches the cap: rosters stay exactly as before.
    assert [s.ship_type for s in auto_draft("alpha", 100, 8, seed=2).ships] == [
        "destroyer", "destroyer", "cruiser", "frigate", "destroyer", "frigate"]


def test_left_battle_torpedo_does_not_enter_captain_shot_history(fleet):
    from src.simulation import SimulationEvent, SimulationEventType
    runner = make_runner(fleet, ["corvette"], ["destroyer"])
    captain = runner.get_captain_for_ship("alpha_1")
    before = len(captain.shot_history)
    runner._feed_captain_event(SimulationEvent(
        event_type=SimulationEventType.PROJECTILE_MISS, timestamp=800.0,
        ship_id="alpha_1", target_id="beta_1",
        data={"projectile_id": "torp_x", "type": "torpedo", "reason": "left_battle",
              "closest_approach_km": 9000.0, "hit_probability": 0.0}))
    assert len(captain.shot_history) == before


# ---------------------------------------------------------------------------
# The torpedo forecast shadow (fire_solutions.fly_torpedo) narrated into the
# battle log: its throw-away 'target' printed MODULE DESTROYED / MORTALLY
# WOUNDED ~70 times per heuristic battle log. Shadows now run with echo off,
# and they never touch the real battle's events or random streams.
# ---------------------------------------------------------------------------

def test_torpedo_forecast_shadow_is_silent_and_isolated(capsys):
    from src.llm.fire_solutions import fly_torpedo
    from src.physics import Vector3D
    from src.simulation import CombatSimulation, create_ship_from_fleet_data
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=7)
    sh = create_ship_from_fleet_data("a1", "corvette", "alpha", data, Vector3D(0, 0, 0),
                                     Vector3D(0, 0, 0), Vector3D(1, 0, 0))
    tg = create_ship_from_fleet_data("b1", "frigate", "beta", data, Vector3D(150_000, 0, 0),
                                     Vector3D(0, 0, 0), Vector3D(1, 0, 0))  # tail-on: the hit wrecks a module
    sim.add_ship(sh)
    sim.add_ship(tg)
    rng_state, dmg_state = sim.rng.getstate(), sim._damage_rng.getstate()
    n_events = len(sim.events)
    capsys.readouterr()
    shots = [fly_torpedo(sim, sh, tg, data, target_evades=False) for _ in range(3)]
    assert any(s is not None and s.hits for s in shots)
    assert capsys.readouterr().out == ""
    assert len(sim.events) == n_events
    assert sim.rng.getstate() == rng_state
    assert sim._damage_rng.getstate() == dmg_state
    assert not sim.torpedoes and sh.torpedoes_remaining == sh.torpedo_launcher.magazine_capacity


def test_fast_spin_on_rcs_overshoots_instead_of_stopping_dead():
    import math
    from src.physics import Vector3D
    sim, ship = _battleship_sim()
    for _ in range(10):
        sim._rotate_ship_toward(ship, Vector3D(0, 1, 0), 1.0, engines_on=True)
    rate = ship.angular_velocity_rad_s
    ahead = ship.forward.rotate_around_axis(Vector3D(0, 0, 1), math.radians(0.5))
    sim._rotate_ship_toward(ship, ahead, 1.0, engines_on=False)
    rcs_alpha = math.radians(ship.attitude_control.rcs_angular_accel_deg_s2)
    assert ship.angular_velocity_rad_s >= rate - 1.01 * rcs_alpha
    assert ship.forward.angle_to(ahead) > math.radians(1.0)   # swung past


@pytest.mark.parametrize("hull", ["corvette", "frigate", "destroyer", "cruiser",
                                  "battleship", "dreadnought"])
@pytest.mark.parametrize("engines_on", [True, False])
def test_90_degree_turn_from_rest_matches_the_hull_rating(hull, engines_on):
    """CLAUDE.md / fleet_ships.json 90-degree turn times, thrust vectoring and
    RCS, flown to a full stop: within a tick (TV) / 2% (RCS). The scalar
    integrator beat them through its free stop at the target (battleship TV
    32 s against 36.9 s)."""
    import math
    from src.physics import Vector3D
    from src.simulation import CombatSimulation, create_ship_from_fleet_data
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    spec = data["ships"][hull]["attitude_control"]
    rated = spec["thrust_vectoring" if engines_on else "rcs"]["time_to_rotate_90_deg_s"]
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=1)
    ship = create_ship_from_fleet_data("s", hull, "alpha", data, Vector3D(0, 0, 0),
                                       Vector3D(0, 0, 0), Vector3D(1, 0, 0))
    sim.add_ship(ship)
    target = Vector3D(0, 1, 0)
    t = 0
    while t < 2000:
        sim._rotate_ship_toward(ship, target, 1.0, engines_on=engines_on)
        t += 1
        if ship.angular_velocity_rad_s == 0.0 and ship.forward.angle_to(target) < 0.01:
            break                                   # on target and stopped
    assert abs(t - rated) <= max(1.5, 0.02 * rated), (t, rated)


# ---------------------------------------------------------------------------
# Evasion respects the turn physics. RUN -> PRESENT assumed the ship could
# flip back instantly; with honest rotation a battleship that turned tail-on
# to RUN from a salvo 70 s out could not turn back and took it in the tail
# (3 tail impacts killed it). Turn times are now flown from the current spin
# (_turn_time_to) and RUN is only started when the round trip fits.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hull", ["corvette", "destroyer", "battleship"])
@pytest.mark.parametrize("spin", ["away", "toward", "side"])
@pytest.mark.parametrize("target_deg", [20, 90, 175])
def test_turn_time_from_a_spin_matches_the_integrator(hull, spin, target_deg):
    import math
    from src.physics import Vector3D
    from src.simulation import CombatSimulation, create_ship_from_fleet_data
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=1)
    ship = create_ship_from_fleet_data("s", hull, "alpha", data, Vector3D(0, 0, 0),
                                       Vector3D(0, 0, 0), Vector3D(1, 0, 0))
    sim.add_ship(ship)
    spin_up = {"away": Vector3D(0, -1, 0), "toward": Vector3D(0, 1, 0),
               "side": Vector3D(0, 0, 1)}[spin]
    for _ in range(8):
        sim._rotate_ship_toward(ship, spin_up, 1.0, engines_on=True)
    assert ship.angular_velocity_rad_s > 0.0
    target = ship.forward.rotate_around_axis(Vector3D(0, 0, 1), math.radians(target_deg))
    predicted = sim._turn_time_to(ship, target, engines_on=True)
    t = 0
    while t < 600:
        sim._rotate_ship_toward(ship, target, 1.0, engines_on=True)
        t += 1
        if ship.forward.angle_to(target) < 0.01 and ship.angular_velocity_rad_s < 1e-3:
            break
    assert 0.75 <= t / max(predicted, 1.0) <= 1.3, (t, predicted)


def _salvo(hull, seed=23, range_km=450):
    import math
    import importlib.util
    from src.simulation import CombatSimulation, SimulationEventType as ET
    spec = importlib.util.spec_from_file_location("tda", "tests/test_doctrine_accuracy.py")
    tda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tda)
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    sim = CombatSimulation(seed=seed, time_step=0.5, decision_interval=1e9)
    sim.echo = False
    victim = tda._build(sim, "victim", hull, "alpha", data, 0, 0, 1)
    tda._evade(victim)
    tda._build(sim, "att", "cruiser_torpedo", "beta", data, range_km, 0, -1)
    launched, modes, struck = 0, [], []
    for _ in range(1600):
        if launched < 6:
            if sim.inject_command("att", {"type": "launch_torpedo", "target_id": "victim"}):
                launched += 1
                continue
        for ev in sim.step():
            if ev.event_type == ET.TORPEDO_IMPACT and ev.data.get("hit_location"):
                struck.append(ev.data["hit_location"])
        if not modes or modes[-1][1] != victim.execution_mode:
            modes.append((sim.current_time, victim.execution_mode))
        if launched >= 6 and not sim.torpedoes:
            break
    return victim, modes, struck


def test_slow_hull_presents_its_nose_instead_of_starting_a_flip_it_cannot_finish():
    """A battleship has ~70 s against a salvo launched 450 km out: a 180 deg
    flip to RUN and back takes ~110 s on thrust vectoring. It holds its
    (thickest) nose on the salvo and every round strikes the nose - the old
    RUN tail-on took the salvo in the tail. (Six rounds still kill it: the
    doctrine's sizing, test_doctrine_accuracy.TestTorpedoSizingDoctrine.)"""
    victim, modes, struck = _salvo("battleship")
    # (Just off the rail the rounds are not yet inside their no-escape zone,
    # so it may start to run for a few seconds - a few degrees of swing.)
    present_at = min(t for t, m in modes if m == "PRESENT")
    assert present_at <= 10.0, modes
    assert all(m == "PRESENT" for t, m in modes if t >= present_at)
    assert len(struck) >= 4 and set(struck) == {"nose"}, struck


def test_fast_hull_still_runs_and_turns_back_before_impact():
    """A frigate can afford the round trip: it runs first, then presents,
    and the salvo strikes the nose."""
    victim, modes, struck = _salvo("frigate")
    names = [m for _, m in modes]
    assert names[0] == "RUN", modes
    run_until = min(t for t, m in modes if m == "PRESENT")
    assert run_until >= 15.0, modes            # a real run, not a twitch
    assert struck and set(struck) == {"nose"}, struck


# A ship spun out of the turn's plane and then ordered far round: the
# estimate used to add the sideways swing as hypot(theta, w_perp^2 / 2a)
# (cruiser 21 s against 38 s stepped, battleship 29 s against 52 s), and the
# integrator itself kept a spin axis whose swing passed 60 deg from the
# target. Now the integrator keeps a spin's axis only while its swing passes
# within SPIN_AXIS_KEEP_DEG of the target, and _turn_time_to brakes the
# sideways spin first and flies the turn from where that leaves the nose.
@pytest.mark.parametrize("hull", ["corvette", "destroyer", "cruiser", "battleship", "dreadnought"])
@pytest.mark.parametrize("target_deg", [60, 120, 170])
def test_turn_time_after_a_sideways_spin_to_a_far_target(hull, target_deg):
    import math
    from src.physics import Vector3D
    from src.simulation import CombatSimulation, create_ship_from_fleet_data
    with open("data/fleet_ships.json") as f:
        data = json.load(f)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=1)
    ship = create_ship_from_fleet_data("s", hull, "alpha", data, Vector3D(0, 0, 0),
                                       Vector3D(0, 0, 0), Vector3D(1, 0, 0))
    sim.add_ship(ship)
    for _ in range(6):
        sim._rotate_ship_toward(ship, Vector3D(0, 0, 1), 1.0, engines_on=True)
    target = Vector3D(math.cos(math.radians(target_deg)), math.sin(math.radians(target_deg)), 0)
    predicted = sim._turn_time_to(ship, target, engines_on=True)
    t = 0
    while t < 600:
        sim._rotate_ship_toward(ship, target, 1.0, engines_on=True)
        t += 1
        if ship.forward.angle_to(target) < 0.01 and ship.angular_velocity_rad_s < 1e-3:
            break
    assert 0.75 <= t / max(predicted, 1.0) <= 1.3, (t, predicted)


# ---------------------------------------------------------------------------
# Order-change tick: weapons fire before ships move, so on the tick a new
# HEADING order is applied the target's drive (which the lead reads) still
# shows the LAST order's burn. The forecast used to take that burn as what
# the target will fly. Traced in seed 39 budget 100 (pre-merge engine) at
# t=270: a cruiser leaving a 6.4 m/s^2 burn for a HEADING turn on a token
# throttle; six rounds forecast at 0.72 all passed outside the gate.
# ---------------------------------------------------------------------------

def _switch_tick_rounds(seed, first="burn"):
    import contextlib
    import io
    import math
    import random
    from src.firecontrol import WeaponsCommand, WeaponsOrder
    from src.physics import Vector3D
    from src.simulation import (CombatSimulation, Maneuver, ManeuverType,
                                SimulationEventType as ET, create_ship_from_fleet_data)
    with open("data/fleet_ships.json") as f:
        fleet = json.load(f)
    rng = random.Random(seed)
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=seed,
                           weapon_arcs_enabled=False)
    tgt = create_ship_from_fleet_data("target", "cruiser_torpedo", "beta", fleet,
                                      position=Vector3D(150e3, 0, 0), forward=Vector3D(-1, 0, 0))
    sim.add_ship(tgt)
    sh = create_ship_from_fleet_data("shooter", "battleship", "alpha", fleet,
                                     position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    sim.add_ship(sh)
    sh.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, target_id="target")
    f0 = tgt.forward.rotate_around_axis(Vector3D(0, 0, 1), math.radians(rng.uniform(60, 120)))
    tgt.current_maneuver = Maneuver(ManeuverType.HEADING, 0, duration=1e9, throttle=1.0,
                                    heading_direction={"x": f0.x, "y": f0.y, "z": f0.z})
    shots, out, stale_next = {}, {}, []
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.current_time < 70.0:
            if abs(sim.current_time - 30.0) < 1e-6:
                ang = rng.uniform(60, 150) * rng.choice([-1, 1])
                d = tgt.forward.rotate_around_axis(Vector3D(0, 0, 1), math.radians(ang))
                tgt.current_maneuver = Maneuver(
                    ManeuverType.HEADING, sim.current_time, duration=1e9, throttle=1.0,
                    heading_direction={"x": d.x, "y": d.y, "z": d.z})
                sh.weapons_orders = {s: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, s, "target",
                                                     min_hit_probability=0.0)
                                     for s in sh.weapons if not s.startswith("pd_")}
            elif sim.current_time > 30.0:
                sh.weapons_orders = {}
                if sim.current_time < 32.0:
                    stale_next.append(sim.target_motion(tgt, sh).stale_burn_ms2)
            for ev in sim.step():
                d = ev.data or {}
                pid = d.get("projectile_id")
                if ev.event_type == ET.PROJECTILE_LAUNCHED:
                    shots[pid] = (d["hit_probability"], d["fc"]["stale"])
                elif pid in shots and pid not in out:
                    if ev.event_type == ET.PROJECTILE_IMPACT:
                        out[pid] = 1
                    elif ev.event_type == ET.PROJECTILE_MISS:
                        out[pid] = 0
    return [(shots[p][0], shots[p][1], out[p]) for p in shots if p in out], stale_next


def test_rounds_fired_on_the_order_change_tick_are_forecast_from_the_new_order():
    """A torpedo cruiser at full burn is ordered onto a heading 60-150 deg
    away; a battleship fires every gun on that tick. Its lead still
    extrapolates the full burn the new order has just ended: no round lands
    (seeds 1-40: 0 of 200), where the forecast used to promise 127; it now
    says ~1.5. The stale burn is charged only on that tick."""
    import math
    rounds, stale_next = [], []
    for seed in range(1, 9):
        r, s = _switch_tick_rounds(seed)
        rounds += r
        stale_next += s
    assert len(rounds) >= 30
    hits = sum(h for _, _, h in rounds)
    forecast = sum(p for p, _, _ in rounds)
    assert all(stale > 5.0 for _, stale, _ in rounds)       # ~13 m/s^2 of old burn
    assert abs(hits - forecast) <= 3.0 * math.sqrt(max(forecast, hits, 1.0)) + 2.0, (hits, forecast)
    assert forecast < 0.1 * len(rounds)
    assert stale_next and max(stale_next) == 0.0             # next tick: the drive shows the order
