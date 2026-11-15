"""
The free commanders' "tactical" doctrine (src/llm/heuristic_tactics.py +
HeuristicCaptain._plan_tactical): focus fire, range discipline by hull role,
massed torpedo salvos sized to the target's point defence, self-preservation
and a formation-matched approach. The "legacy" doctrine stays reachable and
unchanged (the fire-control calibration fixture holds its rounds).
"""

import json
from pathlib import Path

import pytest

from src.physics import Vector3D
from src.simulation import CombatSimulation, Maneuver, ManeuverType, create_ship_from_fleet_data
from src.llm import heuristic_tactics as T
from src.llm.captain import LLMCaptainConfig
from src.llm.heuristic_captain import (
    DOCTRINE_LEGACY, DOCTRINE_TACTICAL, HEURISTIC_MODEL, HeuristicCaptain)
from src.llm.prompts import CaptainPersonality

FLEET = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())


def _ship(sim, sid, stype, faction, pos_km, vel_kps=(0.0, 0.0, 0.0), fwd=(1.0, 0.0, 0.0)):
    ship = create_ship_from_fleet_data(
        sid, stype, faction, FLEET,
        position=Vector3D(*(c * 1000.0 for c in pos_km)),
        velocity=Vector3D(*(c * 1000.0 for c in vel_kps)),
        forward=Vector3D(*fwd))
    ship.name = sid
    ship.current_maneuver = Maneuver(ManeuverType.MAINTAIN, start_time=0.0, duration=0.0)
    sim.add_ship(ship)
    return ship


def _captain(stype, doctrine=None, name="me"):
    return HeuristicCaptain(LLMCaptainConfig(
        name="H", ship_name=name, model=HEURISTIC_MODEL,
        personality=CaptainPersonality.BALANCED, ship_type=stype, fleet_data=FLEET),
        doctrine=doctrine)


def _assert_withdraws(m, foe_dir_x=1.0):
    """A withdrawal inside the retreat range: a full burn away from the gun,
    WITHDRAW_ANGLE_DEG off the line (flank, not tail, to his guns)."""
    import math
    assert m.maneuver_type == ManeuverType.HEADING and m.throttle == 1.0
    h = m.heading_direction
    cos_off = -h["x"] * foe_dir_x / math.sqrt(h["x"] ** 2 + h["y"] ** 2 + h["z"] ** 2)
    assert cos_off == pytest.approx(math.cos(math.radians(T.WITHDRAW_ANGLE_DEG)), abs=0.02)


def _maneuver(commands):
    found = [c for c in commands if isinstance(c, Maneuver)]
    assert found, commands
    return found[-1]


# ---------------------------------------------------------------------------
# Roles and thresholds
# ---------------------------------------------------------------------------

def test_hull_roles_come_from_the_armament():
    sim = CombatSimulation(seed=1, time_step=1.0)
    kinds = {t: _ship(sim, t, t, "alpha", (0, 10 * i, 0)) for i, t in enumerate(
        ("corvette", "cruiser_torpedo", "frigate", "destroyer", "battleship"))}
    assert T.role(kinds["corvette"]) == "torpedo"
    assert T.role(kinds["cruiser_torpedo"]) == "torpedo"
    assert T.role(kinds["frigate"]) == "gun"
    assert not T.has_spinal(kinds["frigate"])
    assert T.has_spinal(kinds["destroyer"]) and T.has_spinal(kinds["battleship"])
    assert T.engage_range_km(kinds["destroyer"]) == T.SPINAL_ENGAGE_RANGE_KM
    assert T.engage_range_km(kinds["frigate"]) == T.TURRET_ENGAGE_RANGE_KM
    # Every gun fight happens inside the doctrine's "~150 km at dodgers".
    assert T.TURRET_ENGAGE_RANGE_KM < T.SPINAL_ENGAGE_RANGE_KM < 150.0
    assert T.SPINAL_ENGAGE_RANGE_KM * T.HOLD_BAND_FACTOR < T.TORPEDO_PANIC_RANGE_KM
    assert T.TORPEDO_PANIC_RANGE_KM < T.TORPEDO_STANDOFF_KM < T.TORPEDO_LAUNCH_RANGE_KM


def test_sprint_only_while_the_brake_still_fits():
    a = 2.0 * 9.81
    assert T.sprint_ok(500.0, 60.0, 0.0, a)            # far, at rest: burn
    assert not T.sprint_ok(150.0, 60.0, 1.0, a)        # too close to build speed
    assert not T.sprint_ok(2000.0, 60.0, T.SPRINT_MAX_CLOSING_KPS, a)  # fast enough
    assert not T.sprint_ok(500.0, 60.0, 0.0, 0.0)      # no drive


# ---------------------------------------------------------------------------
# Focus fire
# ---------------------------------------------------------------------------

def test_focus_prefers_the_damaged_enemy_and_is_sticky():
    sim = CombatSimulation(seed=1, time_step=1.0)
    _ship(sim, "a1", "destroyer", "alpha", (0, 0, 0))
    d1 = _ship(sim, "b1", "destroyer", "beta", (200, 0, 0))
    d2 = _ship(sim, "b2", "destroyer", "beta", (210, 20, 0))
    centre = Vector3D(0, 0, 0)
    assert T.pick_focus([d1, d2], centre, None) is d1          # nearer, equal hulls
    # b2's nose armour is gone: markedly easier to kill, the fleet switches...
    d2.armor.get_section("nose").thickness_cm = 0.0
    assert T.focus_priority(d2, centre) > T.focus_priority(d1, centre) * T.FOCUS_SWITCH_MARGIN
    assert T.pick_focus([d1, d2], centre, "b1") is d2
    # ...but once b1 is as breached (and a bit nearer) the fleet stays on b2.
    d1.armor.get_section("nose").thickness_cm = 0.0
    assert T.focus_priority(d1, centre) > T.focus_priority(d2, centre)
    assert T.pick_focus([d1, d2], centre, "b2") is d2
    # A dying focus is dropped at once.
    d2.is_dying = True
    assert T.pick_focus([d1, d2], centre, "b2") is d1


def test_gun_ships_of_a_fleet_shoot_the_same_target():
    sim = CombatSimulation(seed=2, time_step=1.0)
    _ship(sim, "a1", "destroyer", "alpha", (0, -40, 0))
    _ship(sim, "a2", "destroyer", "alpha", (0, 40, 0))
    _ship(sim, "b1", "destroyer", "beta", (300, -45, 0), fwd=(-1, 0, 0))
    _ship(sim, "b2", "destroyer", "beta", (300, 45, 0), fwd=(-1, 0, 0))
    # Each alpha ship's NEAREST enemy differs (b1 vs b2); the fleet focus does not.
    targets = []
    for sid in ("a1", "a2"):
        captain = _captain("destroyer", name=sid)
        captain.decide(sid, sim)
        targets.append(captain.primary_target_id)
    assert targets[0] == targets[1]
    legacy = []
    for sid in ("a1", "a2"):
        captain = _captain("destroyer", DOCTRINE_LEGACY, name=sid)
        captain.decide(sid, sim)
        legacy.append(captain.primary_target_id)
    assert legacy == ["b1", "b2"]


def test_fleet_plan_is_shared_and_order_independent():
    def plan(order):
        sim = CombatSimulation(seed=3, time_step=1.0)
        for i, t in enumerate(("destroyer", "frigate", "corvette")):
            _ship(sim, f"a{i}", t, "alpha", (0, 30 * i, 0))
        for i, t in enumerate(("cruiser", "frigate")):
            _ship(sim, f"b{i}", t, "beta", (400, 30 * i, 0), fwd=(-1, 0, 0))
        out = {}
        for sid in order:
            captain = _captain(sim.get_ship(sid).ship_type, name=sid)
            captain.decide(sid, sim)
            out[sid] = captain.primary_target_id
        p = T.fleet_plan(sim, "alpha")
        return out, (p.focus_id, p.torpedo_target_id, dict(p.launches))

    a, pa = plan(["a0", "a1", "a2"])
    b, pb = plan(["a2", "a1", "a0"])
    assert a == b and pa == pb


# ---------------------------------------------------------------------------
# Range discipline by role
# ---------------------------------------------------------------------------

def test_gun_policy_never_brakes_under_fire():
    a, R = 2.0 * 9.81, T.SPINAL_ENGAGE_RANGE_KM
    assert T.gun_maneuver(500.0, 0.0, R, a) == ("INTERCEPT", {})          # sprint
    assert T.gun_maneuver(400.0, 3.0, R, a) == ("PADLOCK", {})            # fast enough: coast
    # Inside the guns: coast nose-on through the pass while it is closing...
    for closing in (5.0, 1.0):
        assert T.gun_maneuver(150.0, closing, R, a)[0] == "PADLOCK"
    for closing in (5.0, 1.0, -1.0):
        assert T.gun_maneuver(20.0, closing, R, a)[0] == "PADLOCK"
    # ...re-close nose first once the pass has opened the range...
    assert T.gun_maneuver(R * T.HOLD_BAND_FACTOR + 10, -1.0, R, a) == ("INTERCEPT", {})
    # ...and hold the engagement range once the closure is gone.
    assert T.gun_maneuver(R * 1.1, 0.1, R, a) == ("HOLD_RANGE", {"range_km": R})


@pytest.mark.parametrize("range_km", [150.0, 190.0, 201.0, 220.0, 240.0, 300.0])
@pytest.mark.parametrize("closing", [0.0, -0.2, 0.3])
@pytest.mark.parametrize("accel_g", [1.0, 2.0])
def test_gun_ship_never_stalls_outside_the_hold_band(range_km, closing, accel_g):
    # Review round 3, finding 1: between ~81 and ~245 km with no closure every
    # branch returned PADLOCK (engine off) and two destroyers sat at 220 km
    # for the whole battle.
    kind, _ = T.gun_maneuver(range_km, closing, T.SPINAL_ENGAGE_RANGE_KM, accel_g * 9.81)
    assert kind == "INTERCEPT"


def test_destroyers_started_inside_the_dead_zone_close(tmp_path):
    stats, _ = _small_battle(tmp_path, TACTICAL, hulls=("destroyer",), time_limit=330.0,
                             distance_km=220.0)
    assert stats["closest_km"] < 150.0, stats


def test_gun_ship_faces_the_nearest_gun_and_shoots_the_target():
    sim = CombatSimulation(seed=4, time_step=1.0)
    _ship(sim, "me", "destroyer", "alpha", (0, 0, 0), vel_kps=(1.0, 0, 0))
    slow = _ship(sim, "slow", "battleship", "beta", (110, 30, 0), fwd=(-1, 0, 0))
    slow.armor.get_section("nose").thickness_cm = 0.0   # breached: the fleet's kill
    _ship(sim, "gun", "frigate", "beta", (90, -30, 0), fwd=(-1, 0, 0))
    captain = _captain("destroyer")
    m = _maneuver(captain.decide("me", sim))
    assert captain.primary_target_id == "slow"
    assert m.maneuver_type == ManeuverType.PADLOCK and m.target_id == "gun"
    assert m.evade is None                           # no dodge: it would swing the nose


def test_agile_hulls_are_worth_less_as_targets():
    sim = CombatSimulation(seed=4, time_step=1.0)
    boat = _ship(sim, "c", "corvette", "beta", (100, 0, 0))
    big = _ship(sim, "b", "battleship", "beta", (100, 10, 0))
    assert T.hittability(boat) < T.hittability(big)


def test_capital_ships_close_slowly():
    a = 1.0 * 9.81
    assert T.gun_maneuver(500.0, 0.5, 60.0, a)[0] == "INTERCEPT"
    assert T.gun_maneuver(500.0, 1.2, 60.0, a, 30.0, T.CAPITAL_MAX_CLOSING_KPS)[0] == "PADLOCK"


def test_torpedo_boat_holds_its_standoff_from_the_nearest_gun():
    sim = CombatSimulation(seed=5, time_step=1.0)
    _ship(sim, "me", "cruiser_torpedo", "alpha", (0, 0, 0))
    _ship(sim, "boat", "corvette", "beta", (150, 0, 0), fwd=(-1, 0, 0))
    _ship(sim, "gun", "battleship", "beta", (400, 0, 0), fwd=(-1, 0, 0))
    m = _maneuver(_captain("cruiser_torpedo").decide("me", sim))
    assert m.maneuver_type == ManeuverType.HOLD_RANGE
    assert m.target_id == "gun"
    assert m.hold_range_m == pytest.approx(T.TORPEDO_STANDOFF_KM * 1000.0)


def test_crippled_ship_opens_range(monkeypatch):
    sim = CombatSimulation(seed=6, time_step=1.0)
    _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    _ship(sim, "foe", "cruiser", "beta", (80, 0, 0), fwd=(-1, 0, 0))
    monkeypatch.setattr(T, "hull_pct", lambda s: T.CRIPPLED_HULL_PCT - 5.0 if s.ship_id == "me" else 100.0)
    m = _maneuver(_captain("destroyer").decide("me", sim))
    _assert_withdraws(m)
    assert m.evade is None              # the overlay would replace the withdrawal


# ---------------------------------------------------------------------------
# Massed torpedo salvos
# ---------------------------------------------------------------------------

def _salvo_sim(n_boats, target_type="destroyer", escorts=0, boat="cruiser_torpedo", rng_km=350):
    sim = CombatSimulation(seed=7, time_step=1.0)
    for i in range(n_boats):
        _ship(sim, f"a{i}", boat, "alpha", (0, 20 * i, 0))
    _ship(sim, "tgt", target_type, "beta", (rng_km, 0, 0), fwd=(-1, 0, 0))
    for i in range(escorts):
        _ship(sim, f"esc{i}", "frigate", "beta", (rng_km + 10, 20 * (i + 1), 0), fwd=(-1, 0, 0))
    return sim


def test_every_loaded_tube_with_a_shot_fires():
    sim = _salvo_sim(2)
    plan = T.compute_fleet_plan(sim, "alpha", None, FLEET)
    assert plan.torpedo_target_id == "tgt"
    salvo = plan.salvos[0]
    capacity = sum(sim.torpedo_launch_capacity(b) for b in ("a0", "a1"))
    assert salvo.saturating and sum(salvo.rounds.values()) == capacity > salvo.need > 0
    launched = 0
    for sid in ("a0", "a1"):
        cmds = _captain("cruiser_torpedo", name=sid).decide(sid, sim)
        launched += sum(1 for c in cmds if isinstance(c, dict) and c.get("type") == "launch_torpedo")
    assert launched == capacity


def test_a_lone_corvette_still_fires_at_a_covered_group():
    # One tube cannot saturate a destroyer and four PD escorts; holding would
    # gain nothing (reloads do not stack): it fires, flagged as short.
    sim = _salvo_sim(1, escorts=4, boat="corvette")
    plan = T.compute_fleet_plan(sim, "alpha", None, FLEET)
    assert len(plan.salvos) == 1 and not plan.salvos[0].saturating
    assert sum(n for _, n in plan.launches["a0"]) == sim.torpedo_launch_capacity("a0") > 0


def test_torpedoes_prefer_the_big_slow_hull_under_a_thin_umbrella():
    sim = CombatSimulation(seed=7, time_step=1.0)
    _ship(sim, "a0", "cruiser_torpedo", "alpha", (0, 0, 0))
    _ship(sim, "boat", "corvette", "beta", (300, 0, 0), fwd=(-1, 0, 0))
    _ship(sim, "bb", "battleship", "beta", (330, 400, 0), fwd=(-1, 0, 0))   # out of the boat's PD cover
    plan = T.compute_fleet_plan(sim, "alpha", None, FLEET)
    assert plan.salvos[0].target_id == "bb"


def test_a_gun_ship_on_top_of_the_boats_gets_everything():
    sim = _salvo_sim(1, escorts=4, boat="corvette", rng_km=T.TORPEDO_PANIC_RANGE_KM - 20)
    plan = T.compute_fleet_plan(sim, "alpha", None, FLEET)
    assert sum(n for _, n in plan.launches["a0"]) == sim.torpedo_launch_capacity("a0")


def test_out_of_reach_targets_are_not_launched_at():
    sim = _salvo_sim(2, rng_km=T.TORPEDO_LAUNCH_RANGE_KM + 50)
    assert T.compute_fleet_plan(sim, "alpha", None, FLEET).salvos == []


# ---------------------------------------------------------------------------
# Doctrine selection
# ---------------------------------------------------------------------------

def test_doctrine_selection():
    assert _captain("destroyer").doctrine == DOCTRINE_TACTICAL
    assert _captain("destroyer", DOCTRINE_LEGACY).doctrine == DOCTRINE_LEGACY
    with pytest.raises(ValueError):
        _captain("destroyer", "berserk")


def test_runner_builds_each_factions_doctrine():
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner
    cfg = BattleConfig(verbose=False, personality_selection=False, record_battle=False,
                       heuristic_doctrine={"beta": DOCTRINE_LEGACY})
    dummy = LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                             personality=CaptainPersonality.BALANCED)
    runner = LLMBattleRunner(cfg, dummy, dummy, None)
    a = runner._make_captain(LLMCaptainConfig(name="a", ship_name="a", model=HEURISTIC_MODEL), "alpha")
    b = runner._make_captain(LLMCaptainConfig(name="b", ship_name="b", model=HEURISTIC_MODEL), "beta")
    assert (a.doctrine, b.doctrine) == (DOCTRINE_TACTICAL, DOCTRINE_LEGACY)


# ---------------------------------------------------------------------------
# Seeded battle: tactical fleets fight at range instead of merging
# ---------------------------------------------------------------------------

MIXED = ("destroyer", "destroyer", "corvette", "corvette")
GUNS = ("destroyer", "destroyer", "frigate")


def _small_battle(tmp_path, doctrines, hulls=MIXED, seed=21, time_limit=240.0, distance_km=450.0,
                  beta_hulls=None):
    import importlib.util
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.fleet_config import BattleFleetConfig

    ships = [{"ship_type": t, "model": HEURISTIC_MODEL}
             for t in hulls]
    beta_ships = ships if beta_hulls is None else [
        {"ship_type": t, "model": HEURISTIC_MODEL} for t in beta_hulls]
    config = BattleFleetConfig.from_dict({
        "alpha_fleet": {"ships": ships}, "beta_fleet": {"ships": beta_ships},
        "record_battle": True, "record_sim_trace": True, "personality_selection": False,
        "seed": seed, "time_limit_s": time_limit, "initial_distance_km": distance_km,
    })
    bc = BattleConfig.from_fleet_config(config, verbose=False)
    bc.heuristic_doctrine = dict(doctrines)
    bc.recording_dir = str(tmp_path)
    bc.record_predictions = False
    dummy = LLMCaptainConfig("A", "A", model=HEURISTIC_MODEL)
    runner = LLMBattleRunner(bc, dummy, dummy, None, config)
    runner.run_battle(load_fleet_data())
    path = next(p for p in tmp_path.glob("*.json") if not p.name.endswith(".draft.json"))
    spec = importlib.util.spec_from_file_location(
        "tactics_eval", Path(__file__).parent.parent / "scripts" / "tactics_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    recording = json.loads(path.read_text())
    return ev.analyze(str(path)), recording


TACTICAL = {"alpha": DOCTRINE_TACTICAL, "beta": DOCTRINE_TACTICAL}


@pytest.fixture(scope="module")
def tactical_battle(tmp_path_factory):
    return _small_battle(tmp_path_factory.mktemp("tactical"), TACTICAL)


@pytest.fixture(scope="module")
def gun_battle(tmp_path_factory):
    return _small_battle(tmp_path_factory.mktemp("guns"), TACTICAL, hulls=GUNS, time_limit=330.0)


def test_tactical_gun_fleets_make_a_controlled_pass(gun_battle, tmp_path):
    stats, _ = gun_battle
    legacy, _ = _small_battle(tmp_path, {"alpha": DOCTRINE_LEGACY, "beta": DOCTRINE_LEGACY},
                              hulls=GUNS, time_limit=330.0)
    # Legacy fleets charge through each other at ~6 km/s; tactical ones
    # cap the closure and coast through nose-on.
    assert legacy["closest_rel_kps"] > 4.0, legacy
    assert stats["closest_rel_kps"] < 3.0, stats
    assert stats["shots"] > legacy["shots"], (stats, legacy)
    assert stats["shots_eff_share"] > legacy["shots_eff_share"], (stats, legacy)


def test_tactical_torpedo_boats_launch_as_one_salvo(tactical_battle):
    _, recording = tactical_battle
    firsts = {}
    for e in recording["events"]:
        if e["event_type"] == "torpedo_launched":
            firsts.setdefault(e["ship_id"], e["timestamp"])
    for faction in ("alpha", "beta"):
        times = {t for sid, t in firsts.items() if sid.startswith(faction)}
        boats = [sid for sid in firsts if sid.startswith(faction)]
        assert len(boats) == 2, firsts                  # both corvettes fired...
        assert len(times) == 1, firsts                  # ...in the same checkpoint


def test_tactical_battle_is_deterministic(tactical_battle, tmp_path):
    _, first = tactical_battle
    _, again = _small_battle(tmp_path, TACTICAL)

    def signature(rec):
        return [(e["timestamp"], e["event_type"], e["ship_id"]) for e in rec["events"]
                if e["event_type"] in ("shot_fired", "hit", "torpedo_launched", "ship_destroyed")]
    assert signature(first) == signature(again)


def test_the_plan_is_announced_on_the_fleet_net():
    sim = _salvo_sim(2)
    flag = _captain("cruiser_torpedo", name="a0")
    flag.decide("a0", sim)
    kinds = [m["kind"] for m in flag.outbox if m["tool"] == "fleet_signal"]
    assert kinds == ["TARGET_CLAIM", "TOT_SYNC"]
    other = _captain("cruiser_torpedo", name="a1")
    other.decide("a1", sim)
    assert not other.outbox                      # one voice per fleet


def test_battle_records_the_fleet_net_signals(tactical_battle):
    _, recording = tactical_battle
    kinds = {e["data"]["kind"] for e in recording["events"]
             if e["event_type"] == "comms_message" and e["data"]["channel"] == "fleet_net"
             and e["data"]["sender_role"] == "captain"}
    assert {"TARGET_CLAIM", "TOT_SYNC"} <= kinds


# ---------------------------------------------------------------------------
# Battle damage (src/battle_damage.py): wrecked mounts, dead drives, blasts
# ---------------------------------------------------------------------------

def _wreck_guns(ship):
    for ws in ship.weapons.values():
        if "coil" in str(ws.weapon.weapon_type).lower():
            ws.is_operational = False


def test_a_ship_with_every_gun_wrecked_withdraws_and_says_so():
    sim = CombatSimulation(seed=8, time_step=1.0)
    me = _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    _ship(sim, "mate", "destroyer", "alpha", (0, -40, 0))     # still armed: no surrender
    _ship(sim, "foe", "cruiser", "beta", (90, 0, 0), fwd=(-1, 0, 0))
    _wreck_guns(me)
    assert T.role(me) == "unarmed"
    captain = _captain("destroyer")
    m = _maneuver(captain.decide("me", sim))
    _assert_withdraws(m)
    assert [x["kind"] for x in captain.outbox if x["tool"] == "fleet_signal"][0] == "BREAKING_OFF"


def test_a_disarmed_enemy_is_not_worth_the_focus():
    sim = CombatSimulation(seed=8, time_step=1.0)
    _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    hulk = _ship(sim, "b1", "destroyer", "beta", (150, 0, 0), fwd=(-1, 0, 0))
    _ship(sim, "b2", "destroyer", "beta", (190, 0, 0), fwd=(-1, 0, 0))
    centre = Vector3D(0, 0, 0)
    assert T.pick_focus(list(sim.ships.values())[1:], centre, None).ship_id == "b1"
    _wreck_guns(hulk)
    assert T.pick_focus(list(sim.ships.values())[1:], centre, None).ship_id == "b2"


def test_a_crippled_drive_does_not_set_the_fleet_speed(monkeypatch):
    sim = CombatSimulation(seed=8, time_step=1.0)
    _ship(sim, "a1", "destroyer", "alpha", (0, 0, 0))
    _ship(sim, "a2", "frigate", "alpha", (0, 20, 0))
    _ship(sim, "b1", "destroyer", "beta", (500, 0, 0), fwd=(-1, 0, 0))
    healthy = T.compute_fleet_plan(sim, "alpha", None, FLEET).formation_throttle
    assert healthy["a2"] < 1.0                     # the frigate waits for the destroyer
    monkeypatch.setattr(T, "drive_fraction", lambda s: 0.1 if s.ship_id == "a1" else 1.0)
    assert T.compute_fleet_plan(sim, "alpha", None, FLEET).formation_throttle["a2"] == 1.0


def test_ships_burn_away_from_a_dying_hulk():
    sim = CombatSimulation(seed=8, time_step=1.0)
    _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    mate = _ship(sim, "mate", "cruiser", "alpha", (5, 0, 0))
    _ship(sim, "foe", "cruiser", "beta", (200, 0, 0), fwd=(-1, 0, 0))
    assert T.blast_threat(sim, sim.get_ship("me")) is None
    mate.is_dying = True
    assert T.blast_threat(sim, sim.get_ship("me")) is mate
    m = _maneuver(_captain("destroyer").decide("me", sim))
    assert m.maneuver_type == ManeuverType.HEADING
    assert m.heading_direction["x"] < -0.99         # straight away from the hulk


@pytest.mark.parametrize("stype,foe", [("destroyer", "cruiser"), ("cruiser_torpedo", "battleship")])
@pytest.mark.parametrize("rng_km", [290.0, 340.0, 351.0, 360.0, 400.0])
def test_a_crippled_ship_stays_withdrawn_at_any_range(monkeypatch, stype, foe, rng_km):
    # Review round 3, finding 4: beyond 350 km a crippled gun ship fell through
    # to the attack branch and sprinted back in (a torpedo cruiser to its
    # 300 km standoff), right after signalling BREAKING_OFF.
    sim = CombatSimulation(seed=6, time_step=1.0)
    _ship(sim, "me", stype, "alpha", (0, 0, 0))
    _ship(sim, "foe", foe, "beta", (rng_km, 0, 0), fwd=(-1, 0, 0))
    monkeypatch.setattr(T, "hull_pct", lambda s: T.CRIPPLED_HULL_PCT - 5.0 if s.ship_id == "me" else 100.0)
    m = _maneuver(_captain(stype).decide("me", sim))
    if rng_km < T.RETREAT_RANGE_KM:
        _assert_withdraws(m)
    else:
        assert m.maneuver_type == ManeuverType.HOLD_RANGE and m.target_id == "foe"
        assert m.hold_range_m >= rng_km * 1000.0 - 1.0


def test_a_disarmed_ship_under_fire_actually_opens_range(tmp_path, monkeypatch):
    # Review round 3, finding 2: HOLD_RANGE 350 with evade 'full' was flown as
    # the evasion overlay while he shot at us, and the range kept closing.
    from src.llm.battle_runner import LLMBattleRunner
    orig = LLMBattleRunner.setup_fleet_battle

    def setup(self, fleet_data):
        orig(self, fleet_data)
        _wreck_guns(self.simulation.get_ship("alpha_1"))

    monkeypatch.setattr(LLMBattleRunner, "setup_fleet_battle", setup)
    # A destroyer (2 g) disarmed at t=0 withdraws from a cruiser (1.5 g) that
    # can shoot but not catch it.
    # (An armed frigate keeps alpha in the fight, so it does not surrender.)
    _, rec = _small_battle(tmp_path, TACTICAL, hulls=("destroyer", "frigate"),
                           beta_hulls=("cruiser",), time_limit=240.0, distance_km=150.0)
    frames = rec["sim_trace"]
    import math

    def rng(frame):
        s = frame["ships"]
        return math.dist(s["alpha_1"]["pos"], s["beta_1"]["pos"]) / 1000.0
    shots = [e for e in rec["events"] if e["event_type"] == "shot_fired" and e["ship_id"] == "beta_1"]
    assert shots, "the withdrawing ship must be under fire for this test"
    assert rng(frames[-1]) > rng(frames[30]) + 15.0, (rng(frames[30]), rng(frames[-1]))


def test_a_lone_disarmed_ship_surrenders():
    # Review round 3, finding 3: nothing to shoot with and no armed ship of
    # its own left to protect - it surrenders instead of fleeing forever.
    sim = CombatSimulation(seed=8, time_step=1.0)
    me = _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    _ship(sim, "foe", "destroyer", "beta", (200, 0, 0), vel_kps=(-1.0, 0, 0), fwd=(-1, 0, 0))
    _wreck_guns(me)
    captain = _captain("destroyer")
    captain.decide("me", sim)
    assert captain.has_surrendered            # an equal-drive gun ship is closing in
    # Nobody closing: it withdraws out of gun reach instead (round 5).
    sim3 = CombatSimulation(seed=8, time_step=1.0)
    me3 = _ship(sim3, "me", "destroyer", "alpha", (0, 0, 0))
    _ship(sim3, "foe", "destroyer", "beta", (300, 0, 0), fwd=(-1, 0, 0))
    _wreck_guns(me3)
    captain3 = _captain("destroyer")
    captain3.decide("me", sim3)
    assert not captain3.has_surrendered
    # With an armed friend still fighting it withdraws instead.
    sim2 = CombatSimulation(seed=8, time_step=1.0)
    me2 = _ship(sim2, "me", "destroyer", "alpha", (0, 0, 0))
    _ship(sim2, "mate", "destroyer", "alpha", (0, 30, 0))
    _ship(sim2, "foe", "destroyer", "beta", (300, 0, 0), fwd=(-1, 0, 0))
    _wreck_guns(me2)
    captain2 = _captain("destroyer")
    captain2.decide("me", sim2)
    assert not captain2.has_surrendered


def test_chasers_leave_a_disarmed_runner_for_an_armed_enemy():
    sim = CombatSimulation(seed=8, time_step=1.0)
    me = _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    runner = _ship(sim, "runner", "destroyer", "beta", (260, 0, 0), vel_kps=(2.0, 0, 0))
    _ship(sim, "armed", "cruiser", "beta", (0, 420, 0), fwd=(0, -1, 0))
    _wreck_guns(runner)
    assert T.gun_target(me, [runner, sim.get_ship("armed")], "runner").ship_id == "armed"


def test_the_duel_runner_honours_the_doctrine():
    # Review round 3, finding 6: the 1v1 path built its captains without a faction.
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner
    cfg = BattleConfig(verbose=False, personality_selection=False, record_battle=False,
                       heuristic_doctrine={"alpha": DOCTRINE_LEGACY, "beta": DOCTRINE_TACTICAL})
    runner = LLMBattleRunner(cfg, LLMCaptainConfig("A", "A", model=HEURISTIC_MODEL),
                             LLMCaptainConfig("B", "B", model=HEURISTIC_MODEL), None)
    from src.llm.battle_runner import load_fleet_data
    runner.setup_battle(load_fleet_data())
    assert (runner.alpha_captain.doctrine, runner.beta_captain.doctrine) == (
        DOCTRINE_LEGACY, DOCTRINE_TACTICAL)


def test_fleet_configs_carry_the_heuristic_doctrine():
    # Review round 3, finding 7: --fleet-config / MCP battles had no way to
    # pick the doctrine.
    from src.llm.battle_runner import BattleConfig
    from src.llm.fleet_config import BattleFleetConfig
    ships = [{"ship_type": "destroyer", "model": HEURISTIC_MODEL}]
    both = BattleFleetConfig.from_dict({"alpha_fleet": {"ships": ships}, "beta_fleet": {"ships": ships},
                                        "heuristic_doctrine": "legacy"})
    assert BattleConfig.from_fleet_config(both).heuristic_doctrine == {
        "alpha": DOCTRINE_LEGACY, "beta": DOCTRINE_LEGACY}
    split = BattleFleetConfig.from_dict({"alpha_fleet": {"ships": ships}, "beta_fleet": {"ships": ships},
                                         "heuristic_doctrine": {"beta": "legacy"}})
    assert BattleConfig.from_fleet_config(split).heuristic_doctrine == {"beta": DOCTRINE_LEGACY}
    with pytest.raises(ValueError):
        BattleFleetConfig.from_dict({"alpha_fleet": {"ships": ships}, "beta_fleet": {"ships": ships},
                                     "heuristic_doctrine": "berserk"})


def test_fixture_and_calibration_scripts_default_to_legacy():
    import importlib.util
    import inspect
    root = Path(__file__).parent.parent / "scripts"
    spec = importlib.util.spec_from_file_location("gtb", root / "generate_test_battle.py")
    gtb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gtb)
    assert inspect.signature(gtb.hud_comms_battle).parameters["doctrine"].default == DOCTRINE_LEGACY
    text = (root / "calibrate_torpedo_cost.py").read_text()
    assert '"--doctrine", choices=("legacy", "tactical"), default="legacy"' in text
    assert 'heuristic_doctrine={"alpha": task.get("doctrine", "legacy")' in text


def test_inbound_torpedo_puts_the_evade_overlay_on_the_order(monkeypatch):
    sim = CombatSimulation(seed=9, time_step=1.0)
    _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    _ship(sim, "foe", "destroyer", "beta", (500, 0, 0), fwd=(-1, 0, 0))
    captain = _captain("destroyer")
    assert _maneuver(captain.decide("me", sim)).evade is None
    monkeypatch.setattr(HeuristicCaptain, "_torpedo_threat", lambda self, *a: True)
    m = _maneuver(_captain("destroyer").decide("me", sim))
    assert m.maneuver_type == ManeuverType.INTERCEPT and m.evade == "full"


def test_the_formation_throttle_is_what_the_ship_flies():
    sim = CombatSimulation(seed=9, time_step=1.0)
    _ship(sim, "a1", "destroyer", "alpha", (0, 0, 0))
    _ship(sim, "a2", "frigate", "alpha", (0, 20, 0))
    _ship(sim, "b1", "destroyer", "beta", (500, 0, 0), fwd=(-1, 0, 0))
    m = _maneuver(_captain("frigate", name="a2").decide("a2", sim))
    want = T.fleet_plan(sim, "alpha").formation_throttle["a2"]
    assert want == pytest.approx(2.0 / 3.0, abs=0.01)       # a 3 g frigate waits for a 2 g destroyer
    assert m.maneuver_type == ManeuverType.INTERCEPT and m.throttle == pytest.approx(want)


# Golden: the legacy doctrine is the pass-4 captain bit for bit (the
# fire-control fixture's heuristic battles were flown with it). A change to
# the ENGINE moves this hash too - then re-pin it in the same commit; a
# change that moves it without an engine change has altered legacy.
# Re-pinned for the drive plume (src/drive_plume.py): the 5 km line abreast
# sweeps its torches across its neighbours when it turns, the drive
# interlock holds those burns (`drive_interlock` events, slightly different
# courses) and what wash remains is recorded as `plume_wash`.
LEGACY_GOLDEN_SHA256 = "78a7da477fa1a7fbc990af2a13b3b5c92150dc5961da70c8e0c30bdbfbf9ec99"


def test_legacy_battle_event_stream_is_pinned(tmp_path):
    import hashlib
    _, rec = _small_battle(tmp_path, {"alpha": DOCTRINE_LEGACY, "beta": DOCTRINE_LEGACY},
                           time_limit=150.0)
    stream = [(e["timestamp"], e["event_type"], e["ship_id"],
               json.dumps(e["data"], sort_keys=True, default=str))
              for e in rec["events"] if e["event_type"] not in ("battle_start", "battle_end")]
    digest = hashlib.sha256(json.dumps(stream).encode()).hexdigest()
    assert digest == LEGACY_GOLDEN_SHA256, digest


def test_a_hulk_on_top_of_the_ship_still_gives_a_heading():
    sim = CombatSimulation(seed=8, time_step=1.0)
    _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    _ship(sim, "mate", "cruiser", "alpha", (0, 0, 0)).is_dying = True
    _ship(sim, "foe", "cruiser", "beta", (200, 0, 0), fwd=(-1, 0, 0))
    m = _maneuver(_captain("destroyer").decide("me", sim))
    assert m.maneuver_type == ManeuverType.HEADING
    assert m.heading_direction["x"] < -0.99            # away from the enemy


def test_the_last_ship_of_a_fleet_does_not_signal():
    sim = CombatSimulation(seed=8, time_step=1.0)
    _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    _ship(sim, "foe", "cruiser", "beta", (300, 0, 0), fwd=(-1, 0, 0))
    captain = _captain("destroyer")
    captain.decide("me", sim)
    assert not [m for m in captain.outbox if m["tool"] == "fleet_signal"]


def test_mirror_draft_needs_auto_draft():
    import subprocess
    import sys as _sys
    r = subprocess.run([_sys.executable, "scripts/run_draft_battle.py", "--mirror-draft", "--quiet"],
                       capture_output=True, text=True, cwd=Path(__file__).parent.parent, timeout=120)
    assert r.returncode != 0 and "--mirror-draft needs --auto-draft" in r.stderr


def test_a_lead_with_a_friend_in_its_jet_side_slips_instead_of_idling():
    # Round 4: the plume cap (src/drive_plume.py) idled lead cruisers for whole
    # checkpoints (INTERCEPT at throttle 0) because a follower sat astern.
    sim = CombatSimulation(seed=10, time_step=1.0)
    _ship(sim, "lead", "cruiser", "alpha", (0, 0, 0))
    _ship(sim, "tail", "destroyer", "alpha", (-3, 0, 0))
    _ship(sim, "foe", "cruiser", "beta", (400, 0, 0), fwd=(-1, 0, 0))
    idle = T.plume_safe_throttle(sim, sim.get_ship("lead"), 1.0)
    assert idle < T.PLUME_SLIP_TRIGGER_SHARE            # straight ahead would cook the follower
    m = _maneuver(_captain("cruiser", name="lead").decide("lead", sim))
    assert m.throttle >= T.PLUME_SLIP_MIN_SHARE * T.fleet_plan(sim, "alpha").formation_throttle.get("lead", 1.0)
    if m.maneuver_type == ManeuverType.HEADING:
        import math
        h = m.heading_direction
        assert h["x"] / math.sqrt(h["x"] ** 2 + h["y"] ** 2 + h["z"] ** 2) >= math.cos(math.radians(46))


def test_a_dry_torpedo_fleet_does_not_surrender_with_its_salvo_in_flight(tmp_path):
    # Round 5: six corvettes v a gun line launched all their rounds and
    # surrendered at the next checkpoint with 36 still in flight.
    from src.llm.fleet_draft import auto_draft
    gunline = tuple(s.ship_type for s in auto_draft("beta", budget=150, seed=1, doctrine="gunline").ships)
    _, rec = _small_battle(tmp_path, TACTICAL, hulls=("corvette",) * 6, beta_hulls=gunline,
                           seed=1, time_limit=420.0, distance_km=500.0)
    frames = {round(f["t"]): f for f in rec["sim_trace"]}
    surrenders = [e for e in rec["events"] if e["event_type"] == "surrender"
                  and str(e.get("ship_id") or "").startswith("alpha")]
    for e in surrenders:
        frame = frames.get(round(e["timestamp"])) or frames[max(k for k in frames if k <= e["timestamp"])]
        in_flight = [t for t in frame.get("torpedoes", []) if str(t.get("id", "")).startswith("torp_alpha")]
        assert not in_flight, (e["timestamp"], len(in_flight))
