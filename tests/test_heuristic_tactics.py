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
    # Inside the guns: coast nose-on through the pass, closing or not...
    for closing in (5.0, 1.0, -0.2):
        assert T.gun_maneuver(150.0, closing, R, a)[0] == "PADLOCK"
    for closing in (5.0, 1.0, -1.0):
        assert T.gun_maneuver(20.0, closing, R, a)[0] == "PADLOCK"
    # ...re-close nose first once the pass has opened the range...
    assert T.gun_maneuver(T.REENGAGE_RANGE_KM + 10, -1.0, R, a) == ("INTERCEPT", {})
    # ...and hold the engagement range once the closure is gone.
    assert T.gun_maneuver(R * 1.1, 0.1, R, a) == ("HOLD_RANGE", {"range_km": R})


def test_gun_ship_faces_the_nearest_gun_and_shoots_the_target():
    sim = CombatSimulation(seed=4, time_step=1.0)
    _ship(sim, "me", "destroyer", "alpha", (0, 0, 0), vel_kps=(1.0, 0, 0))
    _ship(sim, "soft", "corvette", "beta", (110, 30, 0), fwd=(-1, 0, 0))
    _ship(sim, "gun", "destroyer", "beta", (90, -30, 0), fwd=(-1, 0, 0))
    captain = _captain("destroyer")
    m = _maneuver(captain.decide("me", sim))
    assert captain.primary_target_id == "soft"       # the fleet's focus (a torpedo boat)
    assert m.maneuver_type == ManeuverType.PADLOCK and m.target_id == "gun"
    assert m.evade is None                           # no dodge: it would swing the nose


def test_torpedo_boat_holds_its_standoff_from_the_nearest_gun():
    sim = CombatSimulation(seed=5, time_step=1.0)
    _ship(sim, "me", "cruiser_torpedo", "alpha", (0, 0, 0))
    _ship(sim, "boat", "corvette", "beta", (150, 0, 0), fwd=(-1, 0, 0))
    _ship(sim, "gun", "battleship", "beta", (400, 0, 0), fwd=(-1, 0, 0))
    m = _maneuver(_captain("cruiser_torpedo").decide("me", sim))
    assert m.maneuver_type == ManeuverType.HOLD_RANGE
    assert m.target_id == "gun"
    assert m.hold_range_m == pytest.approx(T.TORPEDO_STANDOFF_KM * 1000.0)


def test_crippled_ship_opens_range_with_full_evasion(monkeypatch):
    sim = CombatSimulation(seed=6, time_step=1.0)
    _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
    _ship(sim, "foe", "cruiser", "beta", (80, 0, 0), fwd=(-1, 0, 0))
    monkeypatch.setattr(T, "hull_pct", lambda s: T.CRIPPLED_HULL_PCT - 5.0 if s.ship_id == "me" else 100.0)
    m = _maneuver(_captain("destroyer").decide("me", sim))
    assert m.maneuver_type == ManeuverType.HOLD_RANGE
    assert m.target_id == "foe"
    assert m.hold_range_m == pytest.approx(T.RETREAT_RANGE_KM * 1000.0)
    assert m.evade == "full"


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


def test_boats_launch_together_when_the_salvo_saturates_point_defence():
    sim = _salvo_sim(2)
    plan = T.compute_fleet_plan(sim, "alpha", None, FLEET)
    assert plan.torpedo_target_id == "tgt"
    salvo = plan.salvos[0]
    # Sized to his point defence, not every tube on the ship.
    assert sum(salvo.rounds.values()) == salvo.need > 0
    assert salvo.available > salvo.need
    launched = 0
    for sid in ("a0", "a1"):
        captain = _captain("cruiser_torpedo", name=sid)
        cmds = captain.decide(sid, sim)
        launched += sum(1 for c in cmds if isinstance(c, dict) and c.get("type") == "launch_torpedo")
    assert launched == salvo.need


def test_two_boats_pool_their_tubes_for_a_salvo_neither_could_fire_alone():
    sim = _salvo_sim(2, escorts=4)
    plan = T.compute_fleet_plan(sim, "alpha", None, FLEET)
    salvo = plan.salvos[0]
    assert set(salvo.rounds) == {"a0", "a1"}
    assert max(salvo.rounds.values()) < salvo.need <= sum(salvo.rounds.values())


def test_a_lone_corvette_fires_one_best_effort_wave():
    # One tube cannot saturate a destroyer and four PD escorts; holding would
    # gain nothing (reloads do not stack): one wave, flagged as such.
    sim = _salvo_sim(1, escorts=4, boat="corvette")
    plan = T.compute_fleet_plan(sim, "alpha", None, FLEET)
    assert len(plan.salvos) == 1 and not plan.salvos[0].saturating
    assert plan.launches["a0"][1] == 1 < plan.salvos[0].need


def test_salvo_counts_only_rounds_that_leave_together():
    sim = _salvo_sim(1)
    boat = sim.get_ship("a0")
    assert T.tubes_ready_now(sim, boat) == len(boat.ready_torpedo_launchers) == 4
    boat.ready_torpedo_launchers[0].last_launch_time = sim.current_time   # reloading
    assert T.tubes_ready_now(sim, boat) == 3


def test_a_gun_ship_on_top_of_the_boats_gets_everything():
    sim = _salvo_sim(1, escorts=4, boat="corvette", rng_km=T.TORPEDO_PANIC_RANGE_KM - 20)
    plan = T.compute_fleet_plan(sim, "alpha", None, FLEET)
    assert plan.launches["a0"][1] == plan.salvos[0].available > 0


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


def _small_battle(tmp_path, doctrines, hulls=MIXED, seed=21, time_limit=240.0):
    import importlib.util
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.fleet_config import BattleFleetConfig

    ships = [{"ship_type": t, "model": HEURISTIC_MODEL}
             for t in hulls]
    config = BattleFleetConfig.from_dict({
        "alpha_fleet": {"ships": ships}, "beta_fleet": {"ships": ships},
        "record_battle": True, "record_sim_trace": True, "personality_selection": False,
        "seed": seed, "time_limit_s": time_limit, "initial_distance_km": 450.0,
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
    _ship(sim, "foe", "cruiser", "beta", (90, 0, 0), fwd=(-1, 0, 0))
    _wreck_guns(me)
    assert T.role(me) == "unarmed"
    captain = _captain("destroyer")
    m = _maneuver(captain.decide("me", sim))
    assert m.maneuver_type == ManeuverType.HOLD_RANGE
    assert m.hold_range_m == pytest.approx(T.RETREAT_RANGE_KM * 1000.0)
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
