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


@pytest.mark.xfail(strict=False, reason=(
    "Rotation keeps only a scalar |omega| and re-points the rotation axis at "
    "the target every tick, so a turn can reverse instantly. A battleship "
    "spun up on thrust vectoring to 0.047 rad/s and then turning on RCS "
    "(7.5e-5 rad/s^2) toward a point 1 deg behind its nose turns -0.0175 "
    "rad/s the next tick (a 0.064 rad/s change, ~850x its authority) and "
    "stops dead (the overshoot clamp brakes for free). Seen in seed 5 "
    "budget 150: battleship alpha_1 in PADLOCK at t=281-282. Not fixed: "
    "_rotate_ship_toward also flies every evasion jink, and evasion is "
    "pinned by the fire-control calibration fixture. Proposed fix: keep a "
    "signed rate along the previous axis, project it onto the new axis, "
    "brake a negative component at the available authority, and allow "
    "overshoot when the rate exceeds what one tick of braking can stop."))
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

