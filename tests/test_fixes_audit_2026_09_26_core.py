"""
Regression tests for engine-core defects found in the 2026-09-26 audit
(simulation.py / physics / torpedo / point-defense paths).

Each test reproduces one confirmed defect; the docstring states what used to
happen. Assertions are on relationships and invariants, not tuned constants.
"""

import contextlib
import io
import json
import math
from pathlib import Path

import pytest

from src.firecontrol import WeaponsCommand, WeaponsOrder
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    TorpedoInFlight,
    _build_torpedo_specs,
    create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "fleet_ships.json") as f:
    FLEET = json.load(f)

TORPEDO_SPECS = _build_torpedo_specs(FLEET["weapon_types"]["torpedo_launcher"])


def _ship(sim, ship_id, ship_type, faction, position, forward=None):
    ship = create_ship_from_fleet_data(
        ship_id, ship_type, faction, FLEET,
        position=position, forward=forward or Vector3D(1, 0, 0),
    )
    sim.add_ship(ship)
    return ship


def _steps(sim, n):
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            sim.step()


def _events(sim, *types):
    return [e for e in sim.events if e.event_type in types]


def _torpedo(position, velocity, target_id, *, armed=True, fuel_exhausted=False):
    torp = Torpedo(specs=TORPEDO_SPECS, position=position, velocity=velocity,
                   target_id=target_id, guidance_mode=GuidanceMode.COLLISION)
    torp.armed = armed
    torp.fuel_exhausted = fuel_exhausted
    return torp


# ---------------------------------------------------------------------------
# Attitude control: angular rate must not survive while attitude is frozen
# ---------------------------------------------------------------------------

class TestNoStaleAngularVelocity:
    """
    A turn interrupted by a maneuver that does not command attitude (MAINTAIN,
    BRACE, ...) froze the attitude but kept the stored angular rate. The next
    turn then started at that rate: a corvette that had spun up to its 14.8
    deg/s thrust-vectoring limit, coasted, and was then ordered to ROTATE
    (RCS only, 0.12 deg/s^2) finished 90 deg in ~7 s instead of ~50 s.
    """

    def _rcs_turn_time(self, sim, ship, direction):
        sim.inject_command(ship.ship_id, Maneuver(
            ManeuverType.ROTATE, start_time=sim.current_time, direction=direction))
        t0 = sim.current_time
        for _ in range(400):
            _steps(sim, 1)
            if ship.forward.angle_to(direction) < 0.02:
                break
        return sim.current_time - t0

    def _setup(self):
        sim = CombatSimulation(time_step=1.0, seed=1)
        ship = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "corvette", "beta", Vector3D(5e6, 0, 0), Vector3D(-1, 0, 0))
        return sim, ship

    def test_rcs_turn_after_interrupted_tv_turn_is_not_faster_than_from_rest(self):
        sim, ship = self._setup()
        baseline = self._rcs_turn_time(sim, ship, Vector3D(0, 1, 0))

        sim, ship = self._setup()
        sim.inject_command("a", Maneuver(
            ManeuverType.HEADING, start_time=0.0, throttle=1.0,
            heading_direction={"x": -1.0, "y": 0.001, "z": 0.0}))
        _steps(sim, 6)  # spin up on thrust vectoring
        sim.inject_command("a", Maneuver(ManeuverType.MAINTAIN, start_time=sim.current_time))
        frozen = Vector3D(ship.forward.x, ship.forward.y, ship.forward.z)
        _steps(sim, 20)
        # MAINTAIN holds attitude (unchanged behaviour) ...
        assert ship.forward.angle_to(frozen) < 1e-9
        # ... so it must not bank the rate it is no longer turning at.
        assert ship.angular_velocity_rad_s == 0.0

        target = ship.forward.cross(Vector3D(0, 0, 1)).normalized()
        after = self._rcs_turn_time(sim, ship, target)
        assert after >= 0.9 * baseline


# ---------------------------------------------------------------------------
# Surrendered ships are out of the fight
# ---------------------------------------------------------------------------

class TestSurrenderedShipsDoNotFight:
    """
    Surrender (battle_runner sets ship.is_surrendered) documents the ship as
    'drifts and is untargetable', and get_enemy_ships() already hides it. But
    standing weapons orders kept executing in both directions: the
    surrendered ship kept shooting, and enemies kept shooting at it.
    """

    def _duel(self):
        sim = CombatSimulation(time_step=1.0, seed=3)
        _ship(sim, "a", "destroyer", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "destroyer", "beta", Vector3D(300e3, 0, 0), Vector3D(-1, 0, 0))
        for sid, tgt in (("a", "b"), ("b", "a")):
            sim.inject_command(sid, {"type": "set_target", "target_id": tgt})
            sim.inject_command(sid, {"type": "weapons_order", "order": WeaponsOrder(
                weapon_slot="all", command=WeaponsCommand.FIRE_IMMEDIATE, target_id=tgt)})
        return sim

    def test_control_both_sides_fire(self):
        sim = self._duel()
        _steps(sim, 20)
        shooters = {e.ship_id for e in _events(sim, SimulationEventType.PROJECTILE_LAUNCHED)}
        assert shooters == {"a", "b"}

    def test_surrendered_ship_neither_fires_nor_is_fired_upon(self):
        sim = self._duel()
        sim.get_ship("b").is_surrendered = True
        _steps(sim, 20)
        assert _events(sim, SimulationEventType.PROJECTILE_LAUNCHED) == []

    def test_direct_fire_and_torpedo_commands_reject_surrendered_target(self):
        sim = CombatSimulation(time_step=1.0, seed=3)
        _ship(sim, "a", "destroyer", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "t", "corvette", "alpha", Vector3D(0, 5e3, 0))
        _ship(sim, "b", "destroyer", "beta", Vector3D(300e3, 0, 0), Vector3D(-1, 0, 0))
        sim.get_ship("b").is_surrendered = True
        with contextlib.redirect_stdout(io.StringIO()):
            assert not sim.inject_command("a", {
                "type": "fire_at", "weapon_slot": "weapon_1", "target_id": "b"})
            assert not sim.inject_command("t", {"type": "launch_torpedo", "target_id": "b"})
        assert sim.torpedoes == [] and sim.projectiles == []


# ---------------------------------------------------------------------------
# Torpedo retargeting starts a fresh engagement
# ---------------------------------------------------------------------------

class TestRetargetResetsEngagementState:
    """
    When a torpedo's target died, _retarget_torpedo swapped target_id but
    kept the per-engagement state measured against the OLD target:
    min_distance_to_target (closest approach) and the target-side terminal
    latch. The first tick the new target's CPA fell inside the step, the
    torpedo was resolved with the old target's miss distance: a round that
    had passed 150 m from its dead target rolled a 75% hit on a ship still
    120 km away (or, with a km-scale stale value, was discarded as a miss).
    """

    def _scenario(self, stale_min_m):
        sim = CombatSimulation(time_step=1.0, seed=5)
        _ship(sim, "a", "corvette", "alpha", Vector3D(-400e3, 0, 0))
        b1 = _ship(sim, "b1", "destroyer", "beta", Vector3D(0, 0, 0))
        b2 = _ship(sim, "b2", "destroyer", "beta", Vector3D(20e3, 120e3, 0))
        torp = _torpedo(Vector3D(-3000, 0, 0), Vector3D(12000, 0, 0), "b1")
        flight = TorpedoInFlight(torpedo_id="t1", torpedo=torp, source_ship_id="a")
        flight.min_distance_to_target = stale_min_m
        flight.prev_distance_to_target = stale_min_m
        flight.terminal_latched = True
        sim.torpedoes.append(flight)
        b1.is_destroyed = True  # killed by the lead round of the salvo
        return sim, flight, b2

    @pytest.mark.parametrize("stale_min_m", [150.0, 3000.0])
    def test_no_resolution_against_the_new_target_from_long_range(self, stale_min_m):
        sim, flight, b2 = self._scenario(stale_min_m)
        _steps(sim, 4)
        assert _events(sim, SimulationEventType.TORPEDO_RETARGETED)
        assert flight.torpedo.target_id == "b2"
        assert flight.torpedo.position.distance_to(b2.position) > 100e3
        # still flying at the new target, not resolved with the old CPA
        assert flight in sim.torpedoes
        assert not _events(sim, SimulationEventType.TORPEDO_IMPACT)
        assert flight.min_distance_to_target > 50e3

    def test_terminal_latch_does_not_carry_to_new_target(self):
        sim, flight, _ = self._scenario(150.0)
        _steps(sim, 1)
        assert flight.torpedo.target_id == "b2"
        assert flight.terminal_latched is False


# ---------------------------------------------------------------------------
# Torpedo target aspect comes from the approach direction
# ---------------------------------------------------------------------------

class TestTorpedoAspectFromApproachDirection:
    """
    The cross-section a torpedo sees (nose/tail/lateral) was read from the
    bearing between the target and the torpedo's END-of-step position - but
    the terminal encounter is resolved on the step containing the closest
    approach, so that position is already past the hull and the bearing is
    arbitrary. The same nose-on pass at 20 m (destroyer nose radius 15.6 m,
    broadside effective radius 35.3 m) missed when the CPA fell early in the
    1 s step and hit when it fell late.
    """

    def _blinded_pass(self, start_x, miss_m, forward):
        sim = CombatSimulation(time_step=1.0, seed=5)
        _ship(sim, "a", "corvette", "alpha", Vector3D(-900e3, 0, 0))
        _ship(sim, "b", "destroyer", "beta", Vector3D(0, 0, 0), forward)
        torp = _torpedo(Vector3D(start_x, miss_m, 0), Vector3D(-12000, 0, 0), "b",
                        fuel_exhausted=True)
        flight = TorpedoInFlight(torpedo_id="t1", torpedo=torp, source_ship_id="a")
        flight.is_disabled = True
        sim.torpedoes.append(flight)
        _steps(sim, 1)
        return bool(_events(sim, SimulationEventType.TORPEDO_IMPACT))

    @pytest.mark.parametrize("start_x", [1200.0, 6000.0, 11990.0])
    def test_nose_on_pass_outside_nose_radius_misses_regardless_of_step_phase(self, start_x):
        assert not self._blinded_pass(start_x, 20.0, Vector3D(1, 0, 0))

    @pytest.mark.parametrize("start_x", [1200.0, 6000.0, 11990.0])
    def test_broadside_pass_inside_lateral_radius_hits_regardless_of_step_phase(self, start_x):
        # Same trajectory, target turned beam-on: 20 m is inside the
        # broadside effective radius, so it is a hit at any step phase.
        assert self._blinded_pass(start_x, 20.0, Vector3D(0, 0, 1))


# ---------------------------------------------------------------------------
# Torpedo guidance at the live time step (dt = 1 s)
# ---------------------------------------------------------------------------

def _set_remaining_dv(torp, dv_kps):
    """Put a torpedo at the mass that leaves exactly dv_kps in the tank."""
    specs = torp.specs
    torp.current_mass_kg = specs.dry_mass_kg * math.exp(dv_kps / specs.exhaust_velocity_kps)
    torp.remaining_delta_v_kps = dv_kps


def _torpedo_vs_still_target(torp_pos, torp_vel, dv_left, dt=1.0):
    sim = CombatSimulation(time_step=dt, seed=1)
    tgt = _ship(sim, "b", "cruiser", "beta", Vector3D(0, 0, 0), Vector3D(0, 1, 0))
    src = _ship(sim, "a", "corvette", "alpha", Vector3D(-900e3, 0, 0))
    tgt.point_defense = []
    src.point_defense = []
    torp = _torpedo(torp_pos, torp_vel, "b")
    _set_remaining_dv(torp, dv_left)
    flight = TorpedoInFlight(torpedo_id="t1", torpedo=torp, source_ship_id="a")
    sim.torpedoes.append(flight)
    return sim, flight, tgt


class TestTerminalGuidanceDoesNotOvercorrect:
    """
    APN commands a = N * ZEM / t_go^2 with t_go floored at one tick. The
    correction is applied as a full-tick impulse, so once t_go < N*dt it
    moves the miss by up to N times what is needed: at dt = 1 s (the live
    step) and late-flight mass (~60 g) a round drifting to an 80 m coasting
    miss was 'corrected' to a 248 m miss, 120 m to 360 m. Guidance must never
    leave a larger miss than doing nothing.
    """

    @pytest.mark.parametrize("drift_ms,t_go0", [(60.0, 0.8), (100.0, 0.8), (150.0, 0.8),
                                                 (100.0, 0.4), (100.0, 1.5)])
    def test_final_tick_correction_never_worse_than_coasting(self, drift_ms, t_go0):
        vc = 13000.0
        sim, flight, _ = _torpedo_vs_still_target(
            Vector3D(-vc * t_go0, 0, 0), Vector3D(vc, drift_ms, 0), dv_left=0.6)
        for _ in range(5):
            _steps(sim, 1)
            if flight not in sim.torpedoes:
                break
        coasting_miss = drift_ms * t_go0
        assert flight.min_distance_to_target <= coasting_miss + 1.0


class TestTerminalBurnKeepsTheReserve:
    """
    The NEZ terminal burn ran a whole tick at full throttle whatever the
    surplus: at late-flight mass one tick is ~0.6 km/s, so it ate the reserve
    it is documented to preserve (0.37 km/s surplus -> reserve cut from 0.73
    to 0.49 km/s). (The cruise-floor burn has the same full-tick overshoot;
    that one is deliberately left alone - see the audit report.)
    """

    def test_terminal_burn_leaves_the_nez_reserve(self):
        from src.torpedo import TorpedoGuidance
        sim, flight, tgt = _torpedo_vs_still_target(
            Vector3D(-50e3, 0, 0), Vector3D(13000, 0, 0), dv_left=1.1)
        nez = TorpedoGuidance().nez_status(flight.torpedo, tgt.position, tgt.velocity)
        assert nez.inside and nez.dv_margin_kps > 0.2  # precondition: terminal burn
        _steps(sim, 1)
        assert flight.torpedo.remaining_delta_v_kps >= nez.required_delta_v_kps - 0.01
        assert flight.torpedo.remaining_delta_v_kps < 1.1  # it did burn


# ---------------------------------------------------------------------------
# Retarget reachability estimate
# ---------------------------------------------------------------------------

class TestInterceptEstimateIsFrameInvariant:
    """
    Torpedo.calculate_intercept divided the target's WORLD-frame travel by
    a RELATIVE closing rate (target motion counted twice) and used +|v_rel|
    as the closure in the not-closing branch. can_intercept - which gates and
    scores torpedo retargeting - then reported a round already on a perfect
    collision course (20 s out) as 12.5 s and 6 km/s short, and the answer
    changed with the reference frame.
    """

    @pytest.mark.parametrize("frame_vx", [0.0, 5000.0, -5000.0])
    def test_collision_course_needs_no_delta_v(self, frame_vx):
        frame = Vector3D(frame_vx, 0, 0)
        torp = _torpedo(Vector3D(0, 0, 0), Vector3D(10000, 0, 0) + frame, "x")
        ok, t_int, fuel_left = torp.can_intercept(Vector3D(200e3, 0, 0), frame)
        assert ok
        assert t_int == pytest.approx(20.0, rel=1e-3)
        assert fuel_left == pytest.approx(torp.remaining_delta_v_kps, abs=1e-3)

    def test_crossing_and_opening_geometry_is_frame_invariant(self):
        results = []
        for frame_vx in (0.0, 5000.0, -5000.0, 20000.0):
            frame = Vector3D(frame_vx, 0, 0)
            for tgt_vel in (Vector3D(0, 3000, 0), Vector3D(4000, 0, 0)):
                torp = _torpedo(Vector3D(0, 0, 0), Vector3D(6000, 0, 0) + frame, "x")
                results.append((tgt_vel.to_tuple(),) + torp.can_intercept(
                    Vector3D(150e3, 20e3, 0), tgt_vel + frame))
        by_geometry = {}
        for key, ok, t_int, fuel_left in results:
            by_geometry.setdefault(key, []).append((ok, round(t_int, 6), round(fuel_left, 6)))
        for outcomes in by_geometry.values():
            assert len(set(outcomes)) == 1, outcomes


# ---------------------------------------------------------------------------
# A dry torpedo cannot home, so it cannot be MORE accurate than a live one
# ---------------------------------------------------------------------------

class TestDryTorpedoIsNotMoreForgiving:
    """
    The terminal encounter scores the exact CPA with a Gaussian guidance
    error: sigma 150 m live, 500 m once the tank is dry. The CPA already
    carries the missing steering, so the wider sigma only made the dry round
    MORE forgiving: at a 400 m CPA it hit 84% of the time versus 13% live.
    """

    def _logged_probability(self, monkeypatch, fuel_exhausted):
        from src.torpedo import GuidanceCommand, TorpedoGuidance
        monkeypatch.setattr(TorpedoGuidance, "update_collision_guidance",
                            lambda self, *a, **k: GuidanceCommand.coast("test: coast"))
        sim, flight, _ = _torpedo_vs_still_target(
            Vector3D(-20e3, 400.0, 0), Vector3D(10000, 0, 0), dv_left=3.0)
        flight.torpedo.fuel_exhausted = fuel_exhausted

        class _AlwaysMiss:
            def random(self):
                return 0.999999
        monkeypatch.setattr(CombatSimulation, "_torpedo_rng", lambda self, tf: _AlwaysMiss())
        _steps(sim, 4)
        misses = [e for e in _events(sim, SimulationEventType.PROJECTILE_MISS)
                  if e.data.get("type") == "torpedo"]
        assert len(misses) == 1
        return misses[0].data["hit_probability"]

    def test_dry_round_probability_not_above_live(self, monkeypatch):
        live = self._logged_probability(monkeypatch, fuel_exhausted=False)
        dry = self._logged_probability(monkeypatch, fuel_exhausted=True)
        assert dry <= live


# ---------------------------------------------------------------------------
# Point defense and evasion must not depend on the reference frame
# ---------------------------------------------------------------------------

def _coasting_threat_torpedo(position, velocity, target_id):
    torp = _torpedo(position, velocity, target_id, fuel_exhausted=True)
    torp.guidance_mode = GuidanceMode.COAST
    flight = TorpedoInFlight(torpedo_id="t1", torpedo=torp, source_ship_id="E")
    # measure dwell energy, not kills
    flight.ELECTRONICS_THRESHOLD_J = 1e30
    flight.WARHEAD_THRESHOLD_J = 1e30
    return flight


def _pd_heat_by_ship(sim):
    heat = {}
    for e in _events(sim, SimulationEventType.PD_ENGAGED):
        heat[e.ship_id] = heat.get(e.ship_id, 0.0) + e.data["heat_delivered_j"]
    return heat


class TestPointDefenseIsGalileanInvariant:
    """
    Two frame errors in the PD loop, both invisible at rest and large for a
    fleet that is moving:
    - escorts decided whether a torpedo threatened an ally from the
      torpedo's WORLD velocity; with a common 30 km/s drift an escort stopped
      listing a torpedo bearing down on its neighbour and delivered 0 J;
    - PD ran after ships had moved but before torpedoes had, so ranges mixed
      two instants (error = ship speed x dt): at -30 km/s drift a torpedo
      240 km out was seen at 270 km, outside the 250 km envelope.
    """

    def _escort_scene(self, drift):
        v = Vector3D(drift, 0, 0)
        sim = CombatSimulation(time_step=1.0, seed=1)
        for sid, faction, pos in (("A", "alpha", Vector3D(0, 20e3, 0)),
                                  ("B", "alpha", Vector3D(0, 0, 0)),
                                  ("E", "beta", Vector3D(2e6, 0, 0))):
            s = create_ship_from_fleet_data(sid, "destroyer", faction, FLEET,
                                            position=pos, velocity=v)
            sim.add_ship(s)
        sim.torpedoes.append(_coasting_threat_torpedo(
            Vector3D(150e3, 0, 0), v + Vector3D(-15000, 0, 0), "B"))
        return sim

    def test_escort_defends_ally_in_any_frame(self):
        heat = []
        for drift in (0.0, 30000.0, -30000.0):
            sim = self._escort_scene(drift)
            _steps(sim, 5)
            heat.append(_pd_heat_by_ship(sim).get("A", 0.0))
        assert heat[0] > 0.0
        for h in heat[1:]:
            assert h == pytest.approx(heat[0], rel=1e-6)

    def test_dwell_energy_over_an_approach_is_frame_invariant(self):
        totals = []
        for drift in (0.0, 30000.0, -30000.0):
            v = Vector3D(drift, 0, 0)
            sim = CombatSimulation(time_step=1.0, seed=1)
            sim.add_ship(create_ship_from_fleet_data(
                "B", "destroyer", "alpha", FLEET, position=Vector3D(0, 0, 0), velocity=v))
            sim.add_ship(create_ship_from_fleet_data(
                "E", "destroyer", "beta", FLEET, position=Vector3D(5e6, 0, 0), velocity=v))
            flight = _coasting_threat_torpedo(Vector3D(240e3, 0, 0), v + Vector3D(-15000, 0, 0), "B")
            sim.torpedoes.append(flight)
            for _ in range(40):
                if flight not in sim.torpedoes:
                    break
                _steps(sim, 1)
            totals.append(flight.heat_absorbed_j)
        assert totals[0] > 0.0
        for total in totals[1:]:
            assert total == pytest.approx(totals[0], rel=1e-6)

    def test_ballistic_threat_energy_is_relative(self):
        from src.projectile import KineticProjectile
        from src.simulation import ProjectileInFlight
        energies = []
        for drift in (0.0, 40000.0, -40000.0):
            v = Vector3D(drift, 0, 0)
            sim = CombatSimulation(time_step=1.0, seed=1)
            ship = create_ship_from_fleet_data("B", "destroyer", "alpha", FLEET,
                                               position=Vector3D(0, 0, 0), velocity=v)
            sim.add_ship(ship)
            proj = KineticProjectile.from_launch(Vector3D(100e3, 0, 0), v, Vector3D(-1, 0, 0), 7.0, 50.0)
            sim.projectiles.append(ProjectileInFlight(
                projectile_id="p1", projectile=proj, source_ship_id="E", target_ship_id="B"))
            energies.append(sim._gather_ballistic_threats(ship)[0]["ke_gj"])
        expected = 0.5 * 50.0 * 7000.0 ** 2 / 1e9
        for ke in energies:
            assert ke == pytest.approx(expected, rel=1e-6)


class TestSurrenderedShipRunsNoPointDefense:
    """A surrendered ship kept its PD turrets firing (for itself and allies)."""

    def test_no_pd_from_surrendered_ship(self):
        sim = CombatSimulation(time_step=1.0, seed=1)
        for sid, faction, pos in (("A", "alpha", Vector3D(0, 0, 0)),
                                  ("B", "alpha", Vector3D(0, 20e3, 0)),
                                  ("E", "beta", Vector3D(3e6, 0, 0))):
            sim.add_ship(create_ship_from_fleet_data(sid, "destroyer", faction, FLEET, position=pos))
        sim.get_ship("B").is_surrendered = True
        sim.torpedoes.append(_coasting_threat_torpedo(
            Vector3D(150e3, 0, 0), Vector3D(-15000, 0, 0), "A"))
        _steps(sim, 5)
        heat = _pd_heat_by_ship(sim)
        assert heat.get("A", 0.0) > 0.0
        assert "B" not in heat


class TestEngineHeatFollowsProducedThrust:
    """
    Engine waste heat (and the drive's reactor claim) followed the REQUESTED
    throttle: a ship with dry tanks on a BURN order gained 0 m/s but heated
    exactly like a full burn (~600 MJ in 10 s vs ~10 MJ coasting).
    """

    def _heat_gain(self, dry, burn):
        sim = CombatSimulation(time_step=1.0, seed=1)
        ship = _ship(sim, "B", "destroyer", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "E", "destroyer", "beta", Vector3D(5e6, 0, 0))
        if dry:
            ship.kinematic_state.propellant_kg = 0.0
            ship.kinematic_state.mass_kg = ship.kinematic_state.dry_mass_kg
        if burn:
            ship.current_maneuver = Maneuver(ManeuverType.BURN, start_time=0.0, duration=1e6,
                                             throttle=1.0, direction=Vector3D(1, 0, 0))
        h0 = ship.thermal_system.heatsink.current_heat_gj
        _steps(sim, 10)
        return ship.thermal_system.heatsink.current_heat_gj - h0

    def test_dry_ship_on_burn_order_heats_like_a_coasting_ship(self):
        coasting = self._heat_gain(dry=False, burn=False)
        burning = self._heat_gain(dry=False, burn=True)
        dry_burn = self._heat_gain(dry=True, burn=True)
        assert burning > 10 * coasting  # control: a real burn is hot
        assert dry_burn == pytest.approx(coasting, rel=0.05)


class TestManeuversIgnoreDeadTargets:
    """
    INTERCEPT (and EVASIVE's WOBBLE fallback) resolved maneuver.target_id
    without checking it was still alive, so a ship kept burning at full
    throttle toward the frozen position of a destroyed ship - or the drifting
    hulk of a dying one - until its next checkpoint. PADLOCK already dropped
    to a coast in the same situation.
    """

    @pytest.mark.parametrize("fate", ["destroyed", "dying"])
    def test_intercept_on_dead_target_does_not_burn(self, fate):
        sim = CombatSimulation(time_step=1.0, seed=1)
        ship = _ship(sim, "a", "destroyer", "alpha", Vector3D(0, 0, 0))
        dead = _ship(sim, "b", "destroyer", "beta", Vector3D(300e3, 50e3, 0))
        _ship(sim, "c", "destroyer", "beta", Vector3D(-400e3, 0, 0))
        setattr(dead, "is_destroyed" if fate == "destroyed" else "is_dying", True)
        if fate == "dying":
            dead.dying_time_remaining = 1e9
        sim.inject_command("a", Maneuver(ManeuverType.INTERCEPT, start_time=0.0,
                                         throttle=1.0, target_id="b"))
        prop0 = ship.kinematic_state.propellant_kg
        _steps(sim, 5)
        assert ship.kinematic_state.propellant_kg == prop0
        assert ship.velocity.magnitude == 0.0

    def test_wobble_falls_back_to_a_live_enemy(self):
        sim = CombatSimulation(time_step=1.0, seed=1)
        ship = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0), Vector3D(0, 1, 0))
        dead = _ship(sim, "b", "destroyer", "beta", Vector3D(0, 300e3, 0))
        _ship(sim, "c", "destroyer", "beta", Vector3D(0, -300e3, 0))
        dead.is_destroyed = True
        sim.inject_command("a", Maneuver(ManeuverType.EVASIVE, start_time=0.0,
                                         throttle=1.0, target_id="b"))
        _steps(sim, 30)
        # heading for the live enemy at -y, not the wreck at +y
        assert ship.velocity.y < 0.0
