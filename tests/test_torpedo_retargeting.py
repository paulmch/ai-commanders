"""
Torpedo retargeting: a live-seeker round whose target dies mid-flight swings
onto the cheapest reachable enemy instead of orphaning - delta-v and terminal
reserve permitting. Blinded seekers and dry tanks still orphan exactly as
before.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.battle_runner import load_fleet_data
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    SimulationEventType,
    create_ship_from_fleet_data,
)


@pytest.fixture(scope="module")
def fleet_data():
    return load_fleet_data()


def _setup(fleet_data, seed=5, secondary_pos=(350_000, 30_000, 0),
           secondary_vel=(0, 0, 0)):
    """
    Shooter at origin, primary target dead ahead at 300 km, secondary enemy
    near the torpedo's flight path. Returns (sim, events).
    """
    random.seed(seed)
    sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=seed)
    shooter = create_ship_from_fleet_data(
        "alpha", "corvette", "alpha", fleet_data,
        position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0),
        forward=Vector3D(1, 0, 0))
    primary = create_ship_from_fleet_data(
        "beta_1", "corvette", "beta", fleet_data,
        position=Vector3D(300_000, 0, 0), velocity=Vector3D(0, 0, 0),
        forward=Vector3D(-1, 0, 0))
    secondary = create_ship_from_fleet_data(
        "beta_2", "corvette", "beta", fleet_data,
        position=Vector3D(*secondary_pos), velocity=Vector3D(*secondary_vel),
        forward=Vector3D(-1, 0, 0))
    for ship in (shooter, primary, secondary):
        ship.point_defense = []  # measuring guidance, not PD attrition
        sim.add_ship(ship)

    events = []
    sim.add_event_callback(lambda e: events.append(e))
    assert sim.inject_command("alpha", {"type": "launch_torpedo",
                                        "target_id": "beta_1"})
    return sim, events


def _retarget_events(events):
    return [e for e in events
            if e.event_type == SimulationEventType.TORPEDO_RETARGETED]


def _run_until(sim, t, max_steps=4000):
    steps = 0
    while sim.current_time < t and sim.torpedoes and steps < max_steps:
        sim.step()
        steps += 1


class TestRetargeting:

    def test_retargets_to_reachable_enemy_and_hits(self, fleet_data):
        sim, events = _setup(fleet_data)
        _run_until(sim, 8.0)
        assert sim.torpedoes, "torpedo should still be in flight"

        sim.get_ship("beta_1").is_destroyed = True
        _run_until(sim, 12.0)

        retargets = _retarget_events(events)
        assert len(retargets) == 1
        assert retargets[0].data["old_target_id"] == "beta_1"
        assert retargets[0].data["new_target_id"] == "beta_2"
        assert sim.torpedoes[0].torpedo.target_id == "beta_2"

        # And the retargeted round must actually finish the job
        steps = 0
        while sim.torpedoes and steps < 4000:
            sim.step()
            steps += 1
        assert sim.metrics.total_torpedo_hits > 0, \
            "retargeted torpedo never landed"

    def test_blinded_round_does_not_retarget(self, fleet_data):
        sim, events = _setup(fleet_data)
        _run_until(sim, 8.0)
        assert sim.torpedoes

        sim.torpedoes[0].is_disabled = True
        sim.get_ship("beta_1").is_destroyed = True
        _run_until(sim, 14.0)

        assert not _retarget_events(events)
        assert not sim.torpedoes or \
            sim.torpedoes[0].torpedo.target_id == "beta_1"

    def test_dry_tank_does_not_retarget(self, fleet_data):
        sim, events = _setup(fleet_data)
        _run_until(sim, 8.0)
        assert sim.torpedoes

        torp = sim.torpedoes[0].torpedo
        torp.remaining_delta_v_kps = 0.0
        torp.fuel_exhausted = True
        sim.get_ship("beta_1").is_destroyed = True
        _run_until(sim, 14.0)

        assert not _retarget_events(events)

    def test_unreachable_candidate_orphans(self, fleet_data):
        # Secondary enemy far off-axis with a hard transverse velocity, and
        # the torpedo's tank nearly dry: no feasible intercept exists.
        sim, events = _setup(
            fleet_data,
            secondary_pos=(300_000, 5_000_000, 0),
            secondary_vel=(0, 8_000, 0))
        _run_until(sim, 8.0)
        assert sim.torpedoes

        sim.torpedoes[0].torpedo.remaining_delta_v_kps = 1.5
        sim.get_ship("beta_1").is_destroyed = True
        _run_until(sim, 14.0)

        assert not _retarget_events(events)
        # Orphan keeps coasting at its old aim point, no crash, no retarget
        if sim.torpedoes:
            assert sim.torpedoes[0].torpedo.target_id == "beta_1"

    def test_terminal_reserve_is_respected(self, fleet_data):
        """Same geometry, two tank states: only the fuller tank retargets."""
        # Tank trimmed to just above the reserve floor: the lateral steer to
        # beta_2 would eat into the terminal reserve, so the round refuses.
        sim, events = _setup(fleet_data)
        _run_until(sim, 8.0)
        assert sim.torpedoes
        reserve = max(
            sim.RETARGET_MIN_RESERVE_KPS,
            sim.torpedoes[0].torpedo.specs.total_delta_v_kps
            * sim.RETARGET_RESERVE_FRACTION)
        sim.torpedoes[0].torpedo.remaining_delta_v_kps = reserve + 0.05
        sim.get_ship("beta_1").is_destroyed = True
        _run_until(sim, 10.0)
        assert not _retarget_events(events), \
            "retargeted despite eating the terminal reserve"

        # Control: identical setup with a healthy tank retargets fine
        sim2, events2 = _setup(fleet_data)
        _run_until(sim2, 8.0)
        assert sim2.torpedoes
        sim2.get_ship("beta_1").is_destroyed = True
        _run_until(sim2, 10.0)
        assert _retarget_events(events2)

    def test_seeker_health_picks_the_imperative(self, fleet_data):
        """
        Same two candidates, two seeker states. A fresh seeker maximizes
        fuel-at-intercept (impact energy), picking the nearly-free downrange
        chase; a PD-singed seeker races its own blindness and takes the
        faster lateral intercept instead.
        """
        def run(heat_fraction):
            random.seed(9)
            sim = CombatSimulation(time_step=0.5, decision_interval=1e9, seed=9)
            shooter = create_ship_from_fleet_data(
                "alpha", "corvette", "alpha", fleet_data,
                position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0),
                forward=Vector3D(1, 0, 0))
            primary = create_ship_from_fleet_data(
                "beta_1", "corvette", "beta", fleet_data,
                position=Vector3D(300_000, 0, 0), velocity=Vector3D(0, 0, 0),
                forward=Vector3D(-1, 0, 0))
            near_lateral = create_ship_from_fleet_data(
                "beta_2", "corvette", "beta", fleet_data,
                position=Vector3D(150_000, 40_000, 0), velocity=Vector3D(0, 0, 0),
                forward=Vector3D(-1, 0, 0))
            far_downrange = create_ship_from_fleet_data(
                "beta_3", "corvette", "beta", fleet_data,
                position=Vector3D(500_000, 0, 0), velocity=Vector3D(0, 0, 0),
                forward=Vector3D(-1, 0, 0))
            for ship in (shooter, primary, near_lateral, far_downrange):
                ship.point_defense = []
                sim.add_ship(ship)
            events = []
            sim.add_event_callback(lambda e: events.append(e))
            assert sim.inject_command("alpha", {"type": "launch_torpedo",
                                                "target_id": "beta_1"})
            _run_until(sim, 8.0)
            assert sim.torpedoes
            tf = sim.torpedoes[0]
            tf.heat_absorbed_j = heat_fraction * tf.ELECTRONICS_THRESHOLD_J
            sim.get_ship("beta_1").is_destroyed = True
            _run_until(sim, 10.0)
            retargets = _retarget_events(events)
            assert len(retargets) == 1
            return retargets[0].data

        fresh = run(heat_fraction=0.0)
        assert fresh["selection_mode"] == "max_energy"
        assert fresh["new_target_id"] == "beta_3", \
            "fresh seeker should take the fuel-cheap downrange chase"

        singed = run(heat_fraction=0.6)
        assert singed["selection_mode"] == "fastest"
        assert singed["new_target_id"] == "beta_2", \
            "singed seeker should race to the nearest intercept"

    def test_surrendered_ships_are_not_acquired(self, fleet_data):
        sim, events = _setup(fleet_data)
        _run_until(sim, 8.0)
        assert sim.torpedoes

        sim.get_ship("beta_2").is_surrendered = True
        sim.get_ship("beta_1").is_destroyed = True
        _run_until(sim, 14.0)

        assert not _retarget_events(events)
