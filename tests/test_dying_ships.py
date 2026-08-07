"""
Tests for the two-stage ship death (death spiral).

A killing blow that does not destroy the reactor no longer removes the ship
instantly: it becomes a combat-dead hulk - untargetable, weapons and PD
offline, torch sputtering, tumbling slowly - until its reactor detonates a
random 0..30 s later. Ordnance already in flight can still strike the hulk,
bringing the detonation forward, and a hit that destroys the reactor
detonates it immediately. A reactor kill on a healthy ship still destroys
it outright, exactly as before.
"""

import json
import math
from pathlib import Path

import pytest

from src.physics import Vector3D
from src.simulation import (
    DYING_HIT_TIME_BURN_S_PER_GJ,
    DYING_MAX_DURATION_S,
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo, TorpedoSpecs

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "fleet_ships.json") as f:
    FLEET = json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ship(sim, ship_id, faction, position, ship_type="destroyer", forward=None):
    ship = create_ship_from_fleet_data(
        ship_id, ship_type, faction, FLEET,
        position=position, forward=forward or Vector3D(1, 0, 0)
    )
    sim.add_ship(ship)
    return ship


def _destroy_modules(ship, module_type):
    """Zero out every module of the given type; returns the modules."""
    modules = ship._get_modules_by_type(module_type)
    assert modules, f"expected {module_type} modules on {ship.ship_id}"
    for m in modules:
        m.health_percent = 0.0
    return modules


def _events_of(sim, event_type):
    return [e for e in sim.events if e.event_type == event_type]


def _enter_death_spiral(sim, ship, attacker_id="attacker"):
    """Kill the ship through its bridge (critical, non-reactor)."""
    _destroy_modules(ship, "bridge")
    sim._check_ship_destroyed(ship, attacker_id)
    assert ship.is_dying
    return ship


# ---------------------------------------------------------------------------
# Entering (or skipping) the death spiral
# ---------------------------------------------------------------------------

def test_non_reactor_kill_enters_death_spiral():
    sim = CombatSimulation(seed=42, time_step=1.0)
    ship = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))

    _destroy_modules(ship, "bridge")
    sim._check_ship_destroyed(ship, "killer")

    assert ship.is_dying
    assert not ship.is_destroyed
    assert 0.0 <= ship.dying_time_remaining <= DYING_MAX_DURATION_S
    assert ship.kill_credit == "killer"
    # The kill is not booked until the reactor actually goes up
    assert "victim" not in sim.metrics.ships_destroyed
    assert len(_events_of(sim, SimulationEventType.SHIP_DYING)) == 1
    assert len(_events_of(sim, SimulationEventType.SHIP_DESTROYED)) == 0
    # Combat state was dropped on entry
    assert ship.current_maneuver is None
    assert ship.weapons_orders == {}
    assert ship.primary_target_id is None


def test_reactor_kill_detonates_immediately():
    sim = CombatSimulation(seed=42, time_step=1.0)
    ship = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))

    _destroy_modules(ship, "reactor")
    sim._check_ship_destroyed(ship, "killer")

    assert ship.is_destroyed
    assert not ship.is_dying
    assert ship.kill_credit == "killer"
    assert "victim" in sim.metrics.ships_destroyed
    events = _events_of(sim, SimulationEventType.SHIP_DESTROYED)
    assert len(events) == 1
    assert events[0].data["cause"] == "reactor_destroyed"


def test_sublethal_damage_does_not_trigger_death_spiral():
    sim = CombatSimulation(seed=42, time_step=1.0)
    ship = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))

    _destroy_modules(ship, "sensor")  # non-critical
    sim._check_ship_destroyed(ship, "killer")

    assert not ship.is_dying
    assert not ship.is_destroyed


# ---------------------------------------------------------------------------
# Untargetability while dying
# ---------------------------------------------------------------------------

def test_dying_ship_is_untargetable():
    sim = CombatSimulation(seed=7, time_step=1.0)
    shooter = _make_ship(sim, "shooter", "alpha", Vector3D(0, 0, 0))
    victim = _make_ship(sim, "victim", "beta", Vector3D(100_000, 0, 0))
    _enter_death_spiral(sim, victim, "shooter")

    # Dropped from target lists (guns, torpedo retargeting, heuristics all
    # select from get_enemy_ships)
    assert victim not in sim.get_enemy_ships("shooter")

    # Direct fire and torpedo commands at the hulk are refused
    assert not sim.inject_command("shooter", {
        "type": "fire_at", "weapon_slot": next(iter(shooter.weapons), "spinal"),
        "target_id": "victim",
    })
    assert not sim.inject_command("shooter", {
        "type": "launch_torpedo", "target_id": "victim",
    })

    # The dying ship itself no longer accepts commands
    assert not sim.inject_command("victim", Maneuver(
        maneuver_type=ManeuverType.MAINTAIN, start_time=sim.current_time,
        duration=10.0,
    ))


# ---------------------------------------------------------------------------
# The countdown and the detonation
# ---------------------------------------------------------------------------

def test_death_spiral_ends_in_reactor_detonation():
    sim = CombatSimulation(seed=3, time_step=1.0)
    _make_ship(sim, "other", "alpha", Vector3D(1_000_000_000, 0, 0))
    victim = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    _enter_death_spiral(sim, victim, "killer")
    fuse_s = victim.dying_time_remaining

    steps = 0
    while not victim.is_destroyed and steps < int(DYING_MAX_DURATION_S) + 2:
        sim.step()
        steps += 1

    assert victim.is_destroyed
    assert not victim.is_dying
    # Detonation lands when the fuse ran out (within one tick)
    assert steps == pytest.approx(math.ceil(fuse_s) if fuse_s > 0 else 1, abs=1)
    assert victim.kill_credit == "killer"
    assert "victim" in sim.metrics.ships_destroyed

    # The reactor went up: modules destroyed and recorded - the replay viewer
    # keys its instant-detonation path off this MODULE_DESTROYED event
    assert all(m.is_destroyed for m in victim._get_modules_by_type("reactor"))
    reactor_events = [
        e for e in _events_of(sim, SimulationEventType.MODULE_DESTROYED)
        if e.ship_id == "victim" and "reactor" in e.data["module_name"].lower()
    ]
    assert reactor_events

    destroyed_events = _events_of(sim, SimulationEventType.SHIP_DESTROYED)
    assert len(destroyed_events) == 1
    assert destroyed_events[0].data["cause"] == "reactor_detonation"
    assert destroyed_events[0].data["killer_id"] == "killer"


def test_dying_ship_tumbles_and_flickers():
    sim = CombatSimulation(seed=11, time_step=1.0)
    _make_ship(sim, "other", "alpha", Vector3D(1_000_000_000, 0, 0))
    victim = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    _enter_death_spiral(sim, victim)
    victim.dying_time_remaining = DYING_MAX_DURATION_S  # hold the fuse open

    initial_forward = victim.forward
    thrusts = []
    forwards = []
    for _ in range(20):
        sim.step()
        thrusts.append(victim.dying_thrust)
        forwards.append(victim.forward)

    # Slow tumble: orientation drifts but stays a unit basis
    final_forward = forwards[-1]
    assert initial_forward.angle_to(final_forward) > math.radians(5)
    assert final_forward.magnitude == pytest.approx(1.0, abs=1e-6)
    assert abs(final_forward.dot(victim.up)) < 1e-6
    # Tumble rate is slow: bounded by the configured max per tick
    per_tick = [forwards[i].angle_to(forwards[i + 1]) for i in range(len(forwards) - 1)]
    assert max(per_tick) < math.radians(6)

    # The torch sputters at a low throttle rather than burning steadily
    assert all(0.0 <= t <= 0.5 for t in thrusts)


# ---------------------------------------------------------------------------
# Ordnance striking the dying hulk
# ---------------------------------------------------------------------------

def _park_torpedo_on(sim, target, source_id="shooter", speed_ms=5_000.0):
    """A coasting torpedo dead on top of the target, about to hit."""
    torp = Torpedo(
        specs=TorpedoSpecs(),
        position=target.position - Vector3D(1_000, 0, 0),
        velocity=Vector3D(speed_ms, 0, 0),
        target_id=target.ship_id,
    )
    torp.armed = True
    flight = TorpedoInFlight(
        torpedo_id="t1", torpedo=torp, source_ship_id=source_id,
        launch_time=sim.current_time,
    )
    sim.torpedoes.append(flight)
    return flight


def test_hit_on_dying_ship_shortens_the_fuse():
    sim = CombatSimulation(seed=5, time_step=1.0)
    shooter = _make_ship(sim, "shooter", "alpha", Vector3D(500_000, 0, 0))
    victim = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    _enter_death_spiral(sim, victim)
    victim.dying_time_remaining = DYING_MAX_DURATION_S

    flight = _park_torpedo_on(sim, victim)
    sim._resolve_torpedo_hit(flight, victim)

    if victim.is_destroyed:
        # The blast reached the reactor - that is the immediate-detonation
        # path, also legal for a hit on a hulk
        events = _events_of(sim, SimulationEventType.SHIP_DESTROYED)
        assert events[-1].data["cause"] == "reactor_destroyed"
    else:
        # ~3 GJ of kinetic impact must have burned fuse time
        expected_burn = 0.5 * TorpedoSpecs().penetrator_mass_kg * 5_000.0**2 / 1e9 \
            * DYING_HIT_TIME_BURN_S_PER_GJ
        assert victim.dying_time_remaining <= DYING_MAX_DURATION_S - expected_burn * 0.9


def test_reactor_hit_during_death_spiral_detonates_immediately():
    sim = CombatSimulation(seed=5, time_step=1.0)
    victim = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    _enter_death_spiral(sim, victim, "first_killer")
    victim.dying_time_remaining = DYING_MAX_DURATION_S

    _destroy_modules(victim, "reactor")
    sim._check_ship_destroyed(victim, "second_shooter")

    assert victim.is_destroyed
    events = _events_of(sim, SimulationEventType.SHIP_DESTROYED)
    assert events[-1].data["cause"] == "reactor_destroyed"
    # Credit stays with whoever landed the killing blow
    assert victim.kill_credit == "first_killer"


# ---------------------------------------------------------------------------
# Torpedo retargeting only after the detonation
# ---------------------------------------------------------------------------

def test_torpedo_keeps_homing_on_dying_target():
    sim = CombatSimulation(seed=9, time_step=1.0)
    _make_ship(sim, "shooter", "alpha", Vector3D(2_000_000, 0, 0))
    victim = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    _enter_death_spiral(sim, victim)
    victim.dying_time_remaining = DYING_MAX_DURATION_S

    torp = Torpedo(
        specs=TorpedoSpecs(),
        position=Vector3D(400_000, 0, 0),
        velocity=Vector3D(-3_000, 0, 0),
        target_id="victim",
    )
    torp.armed = True
    flight = TorpedoInFlight(
        torpedo_id="t1", torpedo=torp, source_ship_id="shooter",
        launch_time=sim.current_time,
    )
    sim.torpedoes.append(flight)

    sim._update_torpedoes(1.0)

    # A dying target is not an orphaned target: the round keeps its lock
    assert torp.target_id == "victim"
    assert not _events_of(sim, SimulationEventType.TORPEDO_RETARGETED)


def test_torpedo_retargets_only_after_detonation():
    sim = CombatSimulation(seed=9, time_step=1.0)
    _make_ship(sim, "shooter", "alpha", Vector3D(2_000_000, 0, 0))
    victim = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    other = _make_ship(sim, "other_beta", "beta", Vector3D(50_000, 0, 0))
    _enter_death_spiral(sim, victim)

    torp = Torpedo(
        specs=TorpedoSpecs(),
        position=Vector3D(400_000, 0, 0),
        velocity=Vector3D(-3_000, 0, 0),
        target_id="victim",
    )
    torp.armed = True
    flight = TorpedoInFlight(
        torpedo_id="t1", torpedo=torp, source_ship_id="shooter",
        launch_time=sim.current_time,
    )
    sim.torpedoes.append(flight)

    # While the target is merely dying: no retarget
    sim._update_torpedoes(1.0)
    assert torp.target_id == "victim"

    # Reactor goes up: the round swings onto the surviving hull
    sim._detonate_reactor(victim)
    assert victim.is_destroyed
    sim._update_torpedoes(1.0)
    assert torp.target_id == other.ship_id
    assert _events_of(sim, SimulationEventType.TORPEDO_RETARGETED)


# ---------------------------------------------------------------------------
# Battle end waits for the fireworks
# ---------------------------------------------------------------------------

def test_battle_runs_until_pending_detonations_resolve():
    sim = CombatSimulation(seed=13, time_step=1.0)
    _make_ship(sim, "winner", "alpha", Vector3D(1_000_000_000, 0, 0))
    victim = _make_ship(sim, "victim", "beta", Vector3D(0, 0, 0))
    sim._running = True
    _enter_death_spiral(sim, victim)

    # Only one faction can still fight, but the hulk hasn't detonated
    sim._check_battle_end()
    assert sim._running

    sim._detonate_reactor(victim)
    sim._check_battle_end()
    assert not sim._running
