"""
Regression tests for audited defects in src/simulation.py (sim-core partition).

Each test targets a specific confirmed defect; see the test docstrings for the
behaviour that used to be wrong. Tests assert relationships and invariants
rather than magic constants wherever possible.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    ProjectileInFlight,
    SimulationEventType,
    TorpedoInFlight,
    _build_torpedo_specs,
    create_ship_from_fleet_data,
)
from src.projectile import KineticProjectile
from src.torpedo import GuidanceMode, Torpedo, TorpedoSpecs
from src.firecontrol import WeaponsCommand, WeaponsOrder

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


def _charge_all_capacitors(ship):
    if ship.power_system:
        for cap in ship.power_system.weapon_capacitors.values():
            cap.current_charge_mj = cap.capacity_mj


def _make_coasting_torpedo(position, velocity, target_id):
    torp = Torpedo(
        specs=TorpedoSpecs(),
        position=position,
        velocity=velocity,
        target_id=target_id,
    )
    torp.armed = True
    torp.fuel_exhausted = True
    torp.guidance_mode = GuidanceMode.COAST
    return torp


def _events_of(sim, event_type):
    return [e for e in sim.events if e.event_type == event_type]


# ---------------------------------------------------------------------------
# Finding 2: torpedo specs must honor the fleet JSON drive data
# ---------------------------------------------------------------------------

def test_torpedo_specs_use_fleet_drive_data():
    specs = _build_torpedo_specs(FLEET["weapon_types"]["torpedo_launcher"])
    assert specs.exhaust_velocity_kps == pytest.approx(8.0)
    assert specs.total_delta_v_kps == pytest.approx(14.0, rel=0.02)
    accel_g = specs.thrust_n / (specs.mass_kg * 9.81)
    assert accel_g == pytest.approx(12.0, rel=0.02)


def test_ship_torpedo_launcher_matches_json():
    sim = CombatSimulation(seed=1)
    ship = _make_ship(sim, "corv", "alpha", Vector3D(0, 0, 0), ship_type="corvette")
    assert ship.torpedo_launcher is not None
    specs = ship.torpedo_launcher.specs
    # Old code built 60 km/s delta-v torpedoes from a 50 km/s exhaust default
    assert specs.total_delta_v_kps < 20.0
    assert specs.exhaust_velocity_kps == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Finding 1: fast torpedoes must not tunnel through the target
# ---------------------------------------------------------------------------

def test_fast_torpedo_does_not_tunnel():
    """A 120 km/s closing torpedo whose true CPA is ~100 m must trigger a
    terminal resolution (hit or miss) instead of sailing through with the
    closest approach recorded as tens of km."""
    sim = CombatSimulation(seed=42, time_step=1.0)
    _make_ship(sim, "tgt", "beta", Vector3D(0, 0, 0))

    # Offset chosen so no tick-boundary sample lands near the CPA: samples
    # occur at x = -130 km, -10 km, +110 km, so the sampled minimum distance
    # is ~10 km while the true closest approach is 100 m.
    torp = _make_coasting_torpedo(
        position=Vector3D(-250_000, 100, 0),   # true CPA = 100 m
        velocity=Vector3D(120_000, 0, 0),      # 120 km/s closing
        target_id="tgt",
    )
    tf = TorpedoInFlight(torpedo_id="t1", torpedo=torp, source_ship_id="shooter")
    sim.torpedoes.append(tf)

    for _ in range(6):
        sim._update_torpedoes(1.0)
        sim.current_time += 1.0
        if tf not in sim.torpedoes:
            break

    assert tf not in sim.torpedoes, "torpedo tunneled: never resolved"
    impacts = _events_of(sim, SimulationEventType.TORPEDO_IMPACT)
    misses = [e for e in _events_of(sim, SimulationEventType.PROJECTILE_MISS)
              if e.data.get("type") == "torpedo"]
    assert impacts or misses, "no hit/miss resolution occurred"
    # The analytic within-step CPA (~0.1 km) must be recorded, not the
    # tick-boundary sample (~10 km with this geometry, which under the old
    # code also collapsed the hit probability)
    assert tf.min_distance_to_target < 1_000.0


def test_torpedo_hit_roll_uses_seeded_rng():
    """Finding 3: identical seeded runs must produce identical torpedo
    outcomes (the roll used the process-global random module)."""
    def run_once():
        sim = CombatSimulation(seed=7, time_step=1.0)
        _make_ship(sim, "tgt", "beta", Vector3D(0, 0, 0))
        torp = _make_coasting_torpedo(
            position=Vector3D(-60_000, 600, 0),
            velocity=Vector3D(15_000, 0, 0),
            target_id="tgt",
        )
        tf = TorpedoInFlight(torpedo_id="t1", torpedo=torp, source_ship_id="s")
        sim.torpedoes.append(tf)
        for _ in range(10):
            sim._update_torpedoes(1.0)
            sim.current_time += 1.0
        return [(e.event_type.name, e.data.get("roll")) for e in sim.events]

    assert run_once() == run_once()


# ---------------------------------------------------------------------------
# Findings 4 / 14: evasion determinism and RNG hygiene
# ---------------------------------------------------------------------------

THREATS = [{"rel_vel": Vector3D(-2000, 300, 0), "urgency": 1.0}]


def test_evasion_direction_deterministic_and_side_effect_free():
    sim1 = CombatSimulation(seed=42)
    sim2 = CombatSimulation(seed=42)
    ship1 = _make_ship(sim1, "alpha_destroyer", "alpha", Vector3D(0, 0, 0))
    _make_ship(sim2, "alpha_destroyer", "alpha", Vector3D(0, 0, 0))

    d1 = sim1._calculate_evasion_direction(ship1, THREATS)
    d1_again = sim1._calculate_evasion_direction(ship1, THREATS)
    # Same cycle -> same jink direction, and repeat calls are idempotent
    assert (d1.x, d1.y, d1.z) == (d1_again.x, d1_again.y, d1_again.z)

    # Calling the (read-only) evasion query must not perturb the main RNG
    # stream: sim2 never called it, yet must draw the same next number.
    assert sim1.rng.random() == sim2.rng.random()


def test_evasion_direction_stable_across_hash_seeds():
    """The old code keyed the jink on hash(ship_id), which is salted per
    process (PYTHONHASHSEED) - the same seeded battle evaded differently on
    every run. Verify two processes with different hash seeds agree."""
    script = (
        "import json;"
        "from src.simulation import CombatSimulation, create_ship_from_fleet_data;"
        "from src.physics import Vector3D;"
        "fd = json.load(open('data/fleet_ships.json'));"
        "sim = CombatSimulation(seed=42);"
        "ship = create_ship_from_fleet_data('alpha_destroyer','destroyer','alpha',fd,"
        "position=Vector3D(0,0,0), forward=Vector3D(1,0,0));"
        "sim.add_ship(ship);"
        "threats = [{'rel_vel': Vector3D(-2000, 300, 0), 'urgency': 1.0}];"
        "d = sim._calculate_evasion_direction(ship, threats);"
        "print(round(d.x, 12), round(d.y, 12), round(d.z, 12))"
    )
    outs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=str(REPO), env=env, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        outs.append(result.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1]


def test_seeded_battle_is_reproducible():
    """Finding 14: run an identical seeded scenario twice in-process and
    compare kinematics and combat outcomes."""
    def run_battle():
        sim = CombatSimulation(seed=13, time_step=1.0)
        a = _make_ship(sim, "a1", "alpha", Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        b = _make_ship(sim, "b1", "beta", Vector3D(150_000, 0, 0), forward=Vector3D(-1, 0, 0))
        a.primary_target_id = "b1"
        b.primary_target_id = "a1"
        for ship, tgt in ((a, "b1"), (b, "a1")):
            _charge_all_capacitors(ship)
            ship.weapons_orders = {
                slot: WeaponsOrder(command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
                                   target_id=tgt, min_hit_probability=0.1)
                for slot in ship.weapons
            }
            ship.current_maneuver = Maneuver(
                maneuver_type=ManeuverType.EVASIVE, start_time=0.0, throttle=0.5,
            )
        for _ in range(40):
            sim.step()
        state = []
        for s in sim.ships.values():
            state.append((s.ship_id, s.position.x, s.position.y, s.position.z,
                          s.velocity.x, s.velocity.y, s.velocity.z,
                          s.damage_taken_gj, s.shots_fired, s.hits_scored))
        return (sim.metrics.total_shots_fired, sim.metrics.total_hits, state)

    assert run_battle() == run_battle()


# ---------------------------------------------------------------------------
# Finding 5: uncharged capacitors must gate firing
# ---------------------------------------------------------------------------

def test_uncharged_capacitor_blocks_fire():
    sim = CombatSimulation(seed=3)
    a = _make_ship(sim, "a1", "alpha", Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    _make_ship(sim, "b1", "beta", Vector3D(120_000, 0, 0), forward=Vector3D(-1, 0, 0))
    assert a.power_system is not None
    a.weapons_orders = {
        "weapon_0": WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE, target_id="b1")
    }

    # Capacitors start empty: no shot
    for cap in a.power_system.weapon_capacitors.values():
        cap.current_charge_mj = 0.0
    sim._process_weapons_orders(a)
    assert a.shots_fired == 0

    # Charged: fires
    _charge_all_capacitors(a)
    sim._process_weapons_orders(a)
    assert a.shots_fired == 1


# ---------------------------------------------------------------------------
# Findings 6 / 9 / 28: closest-approach handling
# ---------------------------------------------------------------------------

def test_tca_is_signed_when_past_closest_approach():
    sim = CombatSimulation(seed=1)
    # Projectile at origin moving +x, target BEHIND it: receding
    tca, dist = sim._calculate_time_to_closest_approach(
        Vector3D(0, 0, 0), Vector3D(1000, 0, 0),
        Vector3D(-1000, 0, 0), Vector3D(0, 0, 0),
    )
    assert tca < 0, "past closest approach must yield negative TCA"
    assert dist == pytest.approx(1000.0)


def test_projectile_hit_resolves_at_physical_impact_time():
    """Findings 6/9: the proximity shortcut used the closest approach over the
    entire future trajectory, resolving hits up to 4 s early. A 20 km shot at
    5 km/s must record ~4 s of flight, not ~0-1 s."""
    sim = CombatSimulation(seed=5, time_step=1.0)
    shooter = _make_ship(sim, "s1", "alpha", Vector3D(-500_000, 0, 0))
    target = _make_ship(sim, "t1", "beta", Vector3D(20_000, 0, 0))
    target.kinematic_state.velocity = Vector3D(0, 0, 0)

    proj = KineticProjectile.from_launch(
        shooter_position=Vector3D(0, 0, 0),
        shooter_velocity=Vector3D(0, 0, 0),
        target_direction=Vector3D(1, 0, 0),
        muzzle_velocity_kps=5.0,
        mass_kg=50.0,
    )
    sim.projectiles.append(ProjectileInFlight(
        projectile_id="p1", projectile=proj, source_ship_id="s1",
        target_ship_id="t1", launch_time=0.0, hit_probability=1.0,
    ))
    for _ in range(10):
        sim.step()
        if not sim.projectiles:
            break

    impacts = _events_of(sim, SimulationEventType.PROJECTILE_IMPACT)
    assert impacts, "projectile never impacted"
    flight = impacts[0].data["flight_time_s"]
    assert flight >= 3.0, f"hit resolved {4.0 - flight:.1f}s early (flight={flight})"


# ---------------------------------------------------------------------------
# Finding 7: realized hits must follow the reported hit probability
# ---------------------------------------------------------------------------

def test_dispersion_roll_converts_probability_into_misses():
    sim = CombatSimulation(seed=11)
    target = _make_ship(sim, "t1", "beta", Vector3D(0, 0, 0))

    def flight(p):
        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D(-10_000, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_direction=Vector3D(1, 0, 0),
            muzzle_velocity_kps=6.0,
            mass_kg=50.0,
        )
        return ProjectileInFlight(
            projectile_id=f"p{p}", projectile=proj, source_ship_id="s1",
            target_ship_id="t1", launch_time=0.0, hit_probability=p,
        )

    # P(hit) = 0: geometry says hit, roll must convert it into a miss
    sim._resolve_projectile_hit_geometric(flight(0.0), target, None)
    assert target.damage_taken_gj == 0.0
    misses = _events_of(sim, SimulationEventType.PROJECTILE_MISS)
    assert misses and misses[0].data["detection"] == "dispersion_roll"

    # P(hit) = 1: always resolves damage
    sim._resolve_projectile_hit_geometric(flight(1.0), target, None)
    assert target.damage_taken_gj > 0.0


# ---------------------------------------------------------------------------
# Findings 8 / 11: impact frame must be target-relative
# ---------------------------------------------------------------------------

def test_impact_uses_target_relative_velocity():
    sim = CombatSimulation(seed=2)
    target = _make_ship(sim, "t1", "beta", Vector3D(0, 0, 0), forward=Vector3D(-1, 0, 0))
    # Target strafing at 4 km/s, nose held toward the shooter (-x)
    target.kinematic_state.velocity = Vector3D(0, 4000, 0)

    # Slug fired with lead: relative velocity is exactly along the nose axis
    proj = KineticProjectile.from_launch(
        shooter_position=Vector3D(-100, 0, 0),
        shooter_velocity=Vector3D(0, 0, 0),
        target_direction=Vector3D(1, 0, 0),
        muzzle_velocity_kps=6.0,
        mass_kg=10.0,
    )
    proj.velocity = Vector3D(6000, 4000, 0)  # includes lead component
    pf = ProjectileInFlight(
        projectile_id="p1", projectile=proj, source_ship_id="s1",
        target_ship_id="t1", launch_time=0.0,
    )
    sim._resolve_projectile_hit(pf, target)

    impacts = _events_of(sim, SimulationEventType.PROJECTILE_IMPACT)
    assert impacts
    # Rest-frame approach is dead along the nose axis -> NOSE, not LATERAL
    assert impacts[0].data["hit_location"] == "nose"
    # Damage from the 6 km/s closing speed, not the 7.2 km/s world-frame speed
    expected_gj = 0.5 * 10.0 * 6000.0 ** 2 / 1e9
    assert target.damage_taken_gj == pytest.approx(expected_gj, rel=1e-6)


# ---------------------------------------------------------------------------
# Finding 10: destroyed weapon modules must disable weapons on all classes
# ---------------------------------------------------------------------------

def test_module_destruction_disables_weapons_on_cruiser():
    sim = CombatSimulation(seed=1)
    ship = _make_ship(sim, "c1", "alpha", Vector3D(0, 0, 0), ship_type="cruiser")
    assert ship.weapons["weapon_1"].is_operational
    sim._disable_weapon_for_module(ship, "Coilgun Battery A")
    assert not ship.weapons["weapon_1"].is_operational


def test_pd_array_destruction_disables_pd_turrets_on_dreadnought():
    sim = CombatSimulation(seed=1)
    ship = _make_ship(sim, "d1", "alpha", Vector3D(0, 0, 0), ship_type="dreadnought")
    assert any(pd.is_operational for pd in ship.point_defense)
    sim._disable_weapon_for_module(ship, "Point Defense Array")
    assert all(not pd.is_operational for pd in ship.point_defense)


def test_destroyer_pd_module_disables_turret_state():
    sim = CombatSimulation(seed=1)
    ship = _make_ship(sim, "dd1", "alpha", Vector3D(0, 0, 0), ship_type="destroyer")
    sim._disable_weapon_for_module(ship, "PD Laser Dorsal")
    # Both the weapon slot AND the PD turret the engagement loop uses
    assert not ship.weapons["pd_0"].is_operational
    assert not ship.point_defense[0].is_operational


# ---------------------------------------------------------------------------
# Findings 16 / 17 / 20: orphaned munitions must be culled
# ---------------------------------------------------------------------------

def test_orphaned_projectile_is_culled():
    sim = CombatSimulation(seed=1)
    _make_ship(sim, "s1", "alpha", Vector3D(0, 0, 0))
    proj = KineticProjectile.from_launch(
        shooter_position=Vector3D(6_000_000, 0, 0),  # 6000 km from everyone
        shooter_velocity=Vector3D(0, 0, 0),
        target_direction=Vector3D(1, 0, 0),
        muzzle_velocity_kps=6.0,
        mass_kg=50.0,
    )
    sim.projectiles.append(ProjectileInFlight(
        projectile_id="p1", projectile=proj, source_ship_id="s1",
        target_ship_id="no_such_ship", launch_time=0.0,
    ))
    sim._update_projectiles(1.0)
    assert not sim.projectiles


def test_disabled_torpedo_is_culled_by_range():
    sim = CombatSimulation(seed=1)
    _make_ship(sim, "t1", "beta", Vector3D(0, 0, 0))
    torp = _make_coasting_torpedo(
        position=Vector3D(11_000_000, 0, 0),  # beyond the 10,000 km cull
        velocity=Vector3D(1000, 0, 0),
        target_id="t1",
    )
    tf = TorpedoInFlight(torpedo_id="t1", torpedo=torp, source_ship_id="s",
                         is_disabled=True)
    sim.torpedoes.append(tf)
    sim._update_torpedoes(1.0)
    assert tf not in sim.torpedoes


# ---------------------------------------------------------------------------
# Finding 19: flat_chipping comes from the weapon data
# ---------------------------------------------------------------------------

def test_projectile_carries_weapon_for_chipping():
    sim = CombatSimulation(seed=4)
    a = _make_ship(sim, "a1", "alpha", Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    b = _make_ship(sim, "b1", "beta", Vector3D(100_000, 0, 0), forward=Vector3D(-1, 0, 0))
    a.power_system = None  # bypass capacitor gate for this unit test
    a.weapons_orders = {
        "weapon_0": WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE, target_id="b1")
    }
    sim._process_weapons_orders(a)
    assert sim.projectiles
    launched = sim.projectiles[-1]
    assert launched.weapon is a.weapons["weapon_0"].weapon
    assert launched.weapon.flat_chipping == pytest.approx(
        FLEET["weapon_types"]["spinal_coiler_mk3"]["flat_chipping"]
    )


# ---------------------------------------------------------------------------
# Findings 21 / 25: multi-turret PD must not double-count kills
# ---------------------------------------------------------------------------

def test_pd_engagement_skips_already_destroyed_torpedo():
    sim = CombatSimulation(seed=1)
    ship = _make_ship(sim, "s1", "alpha", Vector3D(0, 0, 0))
    ship.power_system = None
    torp = _make_coasting_torpedo(
        position=Vector3D(5_000, 0, 0), velocity=Vector3D(-1000, 0, 0), target_id="s1"
    )
    tf = TorpedoInFlight(torpedo_id="tx", torpedo=torp, source_ship_id="enemy")
    tf.heat_absorbed_j = tf.WARHEAD_THRESHOLD_J + 1  # already past kill threshold
    # NOT in sim.torpedoes: an earlier turret destroyed and removed it
    pd = ship.point_defense[0]
    before = sim.metrics.total_torpedo_intercepted
    sim._pd_engage_torpedo(ship, pd, tf, 1.0)
    assert sim.metrics.total_torpedo_intercepted == before
    assert ship.pd_intercepts == 0
    assert pd.cooldown_remaining == 0.0  # turret did not waste a shot


# ---------------------------------------------------------------------------
# Finding 29: PD delivered energy must not shrink with larger time steps
# ---------------------------------------------------------------------------

def test_pd_heat_delivery_is_dt_invariant():
    def heat_with_dt(dt):
        sim = CombatSimulation(seed=1, time_step=dt)
        ship = _make_ship(sim, "s1", "alpha", Vector3D(0, 0, 0))
        ship.power_system = None
        torp = _make_coasting_torpedo(
            position=Vector3D(5_000, 0, 0), velocity=Vector3D(-1000, 0, 0),
            target_id="s1",
        )
        tf = TorpedoInFlight(torpedo_id="tx", torpedo=torp, source_ship_id="enemy")
        sim.torpedoes.append(tf)
        sim._pd_engage_torpedo(ship, ship.point_defense[0], tf, dt)
        return tf.heat_absorbed_j

    h1 = heat_with_dt(1.0)
    h10 = heat_with_dt(10.0)
    assert h1 > 0
    # cooldown_s = 5: with dt=10 the turret fires once per tick, so it must
    # deliver 10 s worth of power, not 5 s (average power dt-invariant)
    assert h10 == pytest.approx(2.0 * h1, rel=1e-6)


# ---------------------------------------------------------------------------
# Finding 22: sensor report for torpedo-armed ships
# ---------------------------------------------------------------------------

def test_sensor_report_with_torpedo_launcher():
    sim = CombatSimulation(seed=1)
    _make_ship(sim, "corv", "alpha", Vector3D(0, 0, 0), ship_type="corvette")
    report = sim.generate_sensor_report("corv")  # raised AttributeError before
    assert "Torpedoes:" in report


# ---------------------------------------------------------------------------
# Finding 24: fuel-exhausted event logged once, not every tick
# ---------------------------------------------------------------------------

def test_fuel_exhausted_event_logged_once():
    sim = CombatSimulation(seed=1, time_step=1.0)
    _make_ship(sim, "t1", "beta", Vector3D(0, 0, 0))
    torp = _make_coasting_torpedo(
        position=Vector3D(-500_000, 0, 0), velocity=Vector3D(10, 0, 0), target_id="t1"
    )
    tf = TorpedoInFlight(torpedo_id="tc", torpedo=torp, source_ship_id="s")
    sim.torpedoes.append(tf)
    for _ in range(10):
        sim._update_torpedoes(1.0)
        sim.current_time += 1.0
    events = _events_of(sim, SimulationEventType.TORPEDO_FUEL_EXHAUSTED)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Finding 27: step() returns its events; decisions trigger at t=0
# ---------------------------------------------------------------------------

def test_step_returns_events_and_t0_decision_point():
    sim = CombatSimulation(seed=1)
    _make_ship(sim, "s1", "alpha", Vector3D(0, 0, 0))
    events = sim.step()
    assert events, "step() returned no events"
    assert any(e.event_type == SimulationEventType.DECISION_POINT_REACHED
               for e in events), "no decision point at t=0"


# ---------------------------------------------------------------------------
# Finding 31: engine waste heat scales with throttle
# ---------------------------------------------------------------------------

def test_engine_heat_scales_with_throttle():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _make_ship(sim, "s1", "alpha", Vector3D(0, 0, 0))
    assert ship.thermal_system is not None
    source = next(s for s in ship.thermal_system.heat_sources if s.name == "engines")

    def heat_at(throttle):
        ship.current_maneuver = Maneuver(
            maneuver_type=ManeuverType.BURN, start_time=sim.current_time,
            throttle=throttle, direction=Vector3D(1, 0, 0),
        )
        sim._update_ship(ship, 1.0)
        return source.heat_generation_kw

    full = heat_at(1.0)
    low = heat_at(0.05)
    assert full > 0
    assert low == pytest.approx(0.05 * full, rel=1e-6)


# ---------------------------------------------------------------------------
# Finding 13: EVASIVE targets get the fire-control evasion penalty
# ---------------------------------------------------------------------------

def test_evasive_maneuver_reduces_firing():
    """The old code compared maneuver_type.name == 'EVADE' (a nonexistent
    member), so the 40% evasion penalty never applied. With a threshold set
    between P(evading) and P(normal), the shooter must fire on a MAINTAIN
    target and hold on an EVASIVE one."""
    def shots_with(maneuver_type):
        sim = CombatSimulation(seed=9)
        a = _make_ship(sim, "a1", "alpha", Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        b = _make_ship(sim, "b1", "beta", Vector3D(150_000, 0, 0), forward=Vector3D(-1, 0, 0))
        a.power_system = None
        b.current_maneuver = Maneuver(
            maneuver_type=maneuver_type, start_time=0.0, throttle=0.0,
        )
        a.weapons_orders = {
            "weapon_0": WeaponsOrder(command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
                                     target_id="b1", min_hit_probability=0.4)
        }
        sim._process_weapons_orders(a)
        return a.shots_fired

    assert shots_with(ManeuverType.MAINTAIN) == 1
    assert shots_with(ManeuverType.EVASIVE) == 0


# ---------------------------------------------------------------------------
# Finding 23: behavioural coverage for the unexercised maneuver types
# ---------------------------------------------------------------------------

def test_maintain_maneuver_coasts():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _make_ship(sim, "s1", "alpha", Vector3D(0, 0, 0))
    ship.kinematic_state.velocity = Vector3D(0, 1000, 0)
    ship.current_maneuver = Maneuver(
        maneuver_type=ManeuverType.MAINTAIN, start_time=0.0,  # default throttle=1.0
    )
    v0 = ship.velocity.magnitude
    p0 = Vector3D(ship.position.x, ship.position.y, ship.position.z)
    for _ in range(10):
        sim._update_ship(ship, 1.0)
    # MAINTAIN means coast: velocity unchanged, position advances by v*t
    assert ship.velocity.magnitude == pytest.approx(v0, rel=1e-9)
    assert ship.position.y - p0.y == pytest.approx(1000.0 * 10, rel=1e-6)


def test_padlock_maneuver_tracks_target():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _make_ship(sim, "s1", "alpha", Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    _make_ship(sim, "t1", "beta", Vector3D(0, 200_000, 0))
    ship.current_maneuver = Maneuver(
        maneuver_type=ManeuverType.PADLOCK, start_time=0.0, target_id="t1",
    )
    target_dir = Vector3D(0, 1, 0)
    angle0 = math.degrees(ship.forward.angle_to(target_dir))
    for _ in range(40):
        sim._update_ship(ship, 1.0)
    angle1 = math.degrees(ship.forward.angle_to(target_dir))
    assert angle0 == pytest.approx(90.0, abs=1.0)
    assert angle1 < 10.0, f"PADLOCK failed to track: still {angle1:.1f} deg off"


def test_heading_maneuver_flies_requested_direction():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _make_ship(sim, "s1", "alpha", Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    ship.current_maneuver = Maneuver(
        maneuver_type=ManeuverType.HEADING, start_time=0.0, throttle=1.0,
        heading_direction={"x": 0, "y": 1, "z": 0},
    )
    for _ in range(60):
        sim._update_ship(ship, 1.0)
    assert ship.forward.dot(Vector3D(0, 1, 0)) > 0.95
    assert ship.velocity.y > 0, "no velocity gained along requested heading"


def test_rotate_maneuver_turns_without_thrust():
    sim = CombatSimulation(seed=1, time_step=1.0)
    ship = _make_ship(sim, "s1", "alpha", Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    ship.kinematic_state.velocity = Vector3D(500, 0, 0)
    ship.current_maneuver = Maneuver(
        maneuver_type=ManeuverType.ROTATE, start_time=0.0,
        direction=Vector3D(0, 1, 0),
    )
    v0 = ship.velocity.magnitude
    angle0 = math.degrees(ship.forward.angle_to(Vector3D(0, 1, 0)))
    for _ in range(60):
        sim._update_ship(ship, 1.0)
    angle1 = math.degrees(ship.forward.angle_to(Vector3D(0, 1, 0)))
    assert angle1 < angle0 - 15.0, "ROTATE did not rotate the ship"
    assert ship.velocity.magnitude == pytest.approx(v0, rel=1e-6), \
        "ROTATE must not thrust"


# ---------------------------------------------------------------------------
# Finding 12: radiator hits must occur through the combat resolver
# ---------------------------------------------------------------------------

def test_radiator_hits_occur_in_projectile_path():
    """CombatResolver was constructed but never called, so radiators were
    indestructible in battle. With the resolver wired in, repeated hits must
    eventually strike a radiator (5% chance retracted, 20% extended)."""
    sim = CombatSimulation(seed=1)
    target = _make_ship(sim, "t1", "beta", Vector3D(0, 0, 0))
    weapon = target.weapons["weapon_0"].weapon
    for i in range(200):
        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D(-1000, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_direction=Vector3D(1, 0, 0),
            muzzle_velocity_kps=9.9,
            mass_kg=88.0,
        )
        pf = ProjectileInFlight(
            projectile_id=f"p{i}", projectile=proj, source_ship_id="s",
            target_ship_id="t1", weapon=weapon,
        )
        sim._resolve_projectile_hit(pf, target)
    rad_events = _events_of(sim, SimulationEventType.RADIATOR_DAMAGED)
    # With p=0.05 per hit, P(zero in 200) ~ 3.5e-5; seeded run gives 9
    assert rad_events, "no radiator was ever hit through the combat path"


# ---------------------------------------------------------------------------
# Finding 30: PD laser power honours the fleet JSON key
# ---------------------------------------------------------------------------

def test_pd_power_read_from_power_draw_mw():
    fleet = json.loads(json.dumps(FLEET))  # deep copy
    fleet["weapon_types"]["pd_laser"]["power_draw_mw"] = 7.5
    ship = create_ship_from_fleet_data(
        "s1", "destroyer", "alpha", fleet, position=Vector3D(0, 0, 0)
    )
    assert ship.point_defense[0].laser.power_mw == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Findings 15 / 18: geometry-less ships must not crash hit paths
# ---------------------------------------------------------------------------

def test_geometryless_target_does_not_crash():
    sim = CombatSimulation(seed=6, time_step=1.0)
    a = _make_ship(sim, "a1", "alpha", Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    b = _make_ship(sim, "b1", "beta", Vector3D(100_000, 0, 0), forward=Vector3D(-1, 0, 0))
    b.geometry = None  # scenarios path builds ships with geometry=None
    a.power_system = None
    a.weapons_orders = {
        "weapon_0": WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE, target_id="b1")
    }
    sim._process_weapons_orders(a)  # used to raise TypeError (beam_m kwarg)
    assert a.shots_fired == 1

    # Torpedo terminal phase against a geometry-less ship (other TypeError site)
    torp = _make_coasting_torpedo(
        position=Vector3D(90_000, 100, 0), velocity=Vector3D(2_000, 0, 0),
        target_id="b1",
    )
    sim.torpedoes.append(TorpedoInFlight(torpedo_id="tg", torpedo=torp,
                                         source_ship_id="a1"))
    for _ in range(15):
        sim._update_torpedoes(1.0)
        sim.current_time += 1.0
