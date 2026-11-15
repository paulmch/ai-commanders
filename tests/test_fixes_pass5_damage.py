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
