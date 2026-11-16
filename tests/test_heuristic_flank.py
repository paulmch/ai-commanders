"""
Experimental flank doctrines (src/llm/heuristic_flank.py): the static wall
and the pincer / envelopment / two-axis torpedo variants. Never the default.
"""

import json
import math
from pathlib import Path

import pytest

from src.physics import Vector3D
from src.simulation import CombatSimulation, Maneuver, ManeuverType, create_ship_from_fleet_data
from src.llm import heuristic_flank as F
from src.llm.captain import LLMCaptainConfig
from src.llm.heuristic_captain import DEFAULT_DOCTRINE, DOCTRINES, HEURISTIC_MODEL, HeuristicCaptain
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


def _captain(stype, doctrine, name):
    return HeuristicCaptain(LLMCaptainConfig(
        name="H", ship_name=name, model=HEURISTIC_MODEL,
        personality=CaptainPersonality.BALANCED, ship_type=stype, fleet_data=FLEET),
        doctrine=doctrine)


def _maneuver(commands):
    found = [c for c in commands if isinstance(c, Maneuver)]
    assert found, commands
    return found[-1]


def _fleets(sim, doctrine_side="alpha"):
    _ship(sim, "a_bb", "battleship", "alpha", (0, 0, 0))
    _ship(sim, "a_ff", "frigate", "alpha", (0, 30, 0))
    _ship(sim, "a_dd", "destroyer", "alpha", (0, -30, 0))
    _ship(sim, "b_bb", "battleship", "beta", (500, 0, 0), fwd=(-1, 0, 0))
    _ship(sim, "b_ff", "frigate", "beta", (500, 30, 0), fwd=(-1, 0, 0))


def test_flank_doctrines_are_selectable_and_never_the_default():
    for name in F.FLANK_DOCTRINES:
        assert name in DOCTRINES
    assert DEFAULT_DOCTRINE not in F.FLANK_DOCTRINES


def test_pincer_groups_fast_hulls_flank_and_heavies_hold():
    sim = CombatSimulation(seed=1, time_step=1.0)
    _fleets(sim)
    friends = [s for s in sim.ships.values() if s.faction == "alpha"]
    groups = F.assign_groups(friends, F.FLANK_DOCTRINES["pincer70"])
    assert groups == {"a_bb": "hold", "a_ff": "flank_a", "a_dd": "flank_a"}
    env = F.assign_groups(friends, F.FLANK_DOCTRINES["envelop70"])
    assert {env["a_ff"], env["a_dd"], env["a_bb"]} == {"flank_a", "flank_b"}
    wall = F.assign_groups(friends, F.FLANK_DOCTRINES["wall"])
    assert set(wall.values()) == {"hold"}


def test_flankers_position_outside_gun_reach_with_the_evade_overlay():
    sim = CombatSimulation(seed=2, time_step=1.0)
    _fleets(sim)
    m = _maneuver(_captain("frigate", "pincer70", "a_ff").decide("a_ff", sim))
    assert m.maneuver_type == ManeuverType.HEADING and m.evade == "dodge"
    st = F.flank_state(sim, "alpha", F.FLANK_DOCTRINES["pincer70"])
    station = st.stations["flank_a"]
    enemy = (sim.get_ship("b_bb").position + sim.get_ship("b_ff").position) * 0.5
    rng = (station - enemy).magnitude / 1000.0
    assert rng == pytest.approx(F.POSITION_RANGE_KM, abs=1.0)
    axis = (sim.get_ship("a_bb").position - enemy).normalized()
    ang = math.degrees(math.acos((station - enemy).normalized().dot(axis)))
    assert ang == pytest.approx(70.0, abs=1.0)
    # The holding battleship waits outside the wall's reach meanwhile.
    m = _maneuver(_captain("battleship", "pincer70", "a_bb").decide("a_bb", sim))
    assert m.maneuver_type in (ManeuverType.INTERCEPT, ManeuverType.PADLOCK)


def test_the_attack_run_starts_when_the_flankers_are_on_station():
    sim = CombatSimulation(seed=3, time_step=1.0)
    _fleets(sim)
    spec = F.FLANK_DOCTRINES["pincer70"]
    st = F.flank_state(sim, "alpha", spec)
    assert not st.attack_started
    station = st.stations["flank_a"]
    for sid in ("a_ff", "a_dd"):
        sim.get_ship(sid).kinematic_state.position = station
    sim.current_time += 30.0
    st = F.flank_state(sim, "alpha", spec)
    assert st.attack_started
    m = _maneuver(_captain("frigate", "pincer70", "a_ff").decide("a_ff", sim))
    assert m.maneuver_type == ManeuverType.INTERCEPT and m.standoff_m is not None


def test_the_wall_closes_slowly_then_coasts_nose_on():
    assert F.wall_maneuver(400.0, 0.0) == "INTERCEPT"
    assert F.wall_maneuver(400.0, F.WALL_CLOSE_KPS + 0.1) == "PADLOCK"
    assert F.wall_maneuver(F.WALL_ENGAGE_KM - 1.0, 0.0) == "PADLOCK"


def test_split_boats_take_two_bearings():
    sim = CombatSimulation(seed=4, time_step=1.0)
    _ship(sim, "a_c1", "corvette", "alpha", (0, 0, 0))
    _ship(sim, "a_c2", "corvette", "alpha", (0, 20, 0))
    _ship(sim, "b_dd", "destroyer", "beta", (400, 0, 0), fwd=(-1, 0, 0))
    m1 = _maneuver(_captain("corvette", "split", "a_c1").decide("a_c1", sim))
    m2 = _maneuver(_captain("corvette", "split", "a_c2").decide("a_c2", sim))
    st = F.flank_state(sim, "alpha", F.FLANK_DOCTRINES["split"])
    enemy = sim.get_ship("b_dd").position
    a, b = st.stations["boat_a"] - enemy, st.stations["boat_b"] - enemy
    sep = math.degrees(math.acos(a.normalized().dot(b.normalized())))
    assert sep == pytest.approx(2 * F.SPLIT_HALF_DEG, abs=1.0)
    assert m1.maneuver_type == ManeuverType.HEADING and m2.maneuver_type == ManeuverType.HEADING
