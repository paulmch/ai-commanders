"""
Maneuver semantics: the maneuver tools mean what commanders think they mean.

Background (pass-4 audit, battle-blunders R3/R4, captain-forensics CP-03/CP-16):

* Plain INTERCEPT accelerates to the end: fleets merged at 3.5-7 km/s in 5 of
  7 recorded battles because nothing ever braked. INTERCEPT now takes a
  standoff range and/or an arrival closing speed: it closes, flips, brakes
  and holds the range. HOLD_RANGE keeps a set range (cancel closure).
* set_heading is a DIRECTION; admirals kept sending POSITIONS ("fly to
  (90,55,10)"), which sent one corvette burning away from the enemy. HEADING
  now also flies toward a point or a ship's CURRENT bearing.
* 28% of admiral orders asked for INTERCEPT and EVADE at once, but a ship has
  one maneuver slot. Any maneuver can now carry the evasion overlay.

Everything new is opt-in: with none of the new fields set, every maneuver is
flown exactly as before (pinned by the seeded battle comparison in the pass-4
report and by tests/test_trajectory_prediction.py).
"""

import json
import math
from pathlib import Path

import pytest

from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    HEADING_POINT_ARRIVAL_M,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    create_ship_from_fleet_data,
    maneuver_reacts_to_threats,
)
from src.maneuvers import max_arrival_speed_mps, required_decel_ms2, stopping_distance_m
from src.replay_evidence import json_value

FLEET = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())


def _ship(sim, sid, stype, faction, pos_km, vel_kps=(0.0, 0.0, 0.0), fwd=(1.0, 0.0, 0.0)):
    ship = create_ship_from_fleet_data(
        sid, stype, faction, FLEET,
        position=Vector3D(*(c * 1000.0 for c in pos_km)),
        velocity=Vector3D(*(c * 1000.0 for c in vel_kps)),
        forward=Vector3D(*fwd))
    ship.name = sid
    sim.add_ship(ship)
    return ship


def _coast(ship):
    ship.current_maneuver = Maneuver(ManeuverType.MAINTAIN, start_time=0.0, duration=0.0)


def _range_m(a, b):
    return (a.position - b.position).magnitude


def _duel(seed=7, stype="destroyer", foe_vel=(0.0, 0.0, 0.0), my_vel=(0.0, 0.0, 0.0),
          foe_km=400.0):
    sim = CombatSimulation(seed=seed, time_step=1.0)
    me = _ship(sim, "me", stype, "alpha", (0, 0, 0), vel_kps=my_vel)
    foe = _ship(sim, "foe", "destroyer", "beta", (foe_km, 0, 0), vel_kps=foe_vel, fwd=(-1, 0, 0))
    _coast(foe)
    return sim, me, foe


def _fly(sim, me, foe, make, seconds, renew_s=None):
    """Fly ``make()`` (a Maneuver factory) for ``seconds``, renewing it every
    ``renew_s`` like a commander at each checkpoint. Returns the closest
    approach (m), the relative speed there (m/s) and the per-second log."""
    sim.inject_command(me.ship_id, make(sim))
    best_r, best_v = float("inf"), None
    log = []
    for k in range(int(seconds)):
        if renew_s and k and k % renew_s == 0:
            sim.inject_command(me.ship_id, make(sim))
        sim.step()
        r = _range_m(me, foe)
        v = (me.velocity - foe.velocity).magnitude
        if r < best_r:
            best_r, best_v = r, v
        log.append((sim.current_time, r, v, me.execution_mode))
    return best_r, best_v, log


def _intercept(standoff_km=None, arrival_kps=None, throttle=1.0, renew=False, evade=None):
    def make(sim):
        return Maneuver(
            ManeuverType.INTERCEPT, start_time=sim.current_time,
            duration=30.0 if renew else 0.0, throttle=throttle, target_id="foe",
            standoff_m=None if standoff_km is None else standoff_km * 1000.0,
            arrival_closing_mps=None if arrival_kps is None else arrival_kps * 1000.0,
            evade=evade)
    return make


def _hold(range_km=None, throttle=1.0, renew=False):
    def make(sim):
        return Maneuver(
            ManeuverType.HOLD_RANGE, start_time=sim.current_time,
            duration=30.0 if renew else 0.0, throttle=throttle, target_id="foe",
            hold_range_m=None if range_km is None else range_km * 1000.0)
    return make


# =============================================================================
# Kinematics helpers (src/maneuvers.py)
# =============================================================================

class TestArrivalKinematics:

    def test_stopping_distance_and_required_decel_are_inverse(self):
        d = stopping_distance_m(2000.0, 16.0, end_speed_mps=300.0)
        assert d == pytest.approx((2000.0 ** 2 - 300.0 ** 2) / 32.0)
        assert required_decel_ms2(2000.0, d, 300.0) == pytest.approx(16.0)
        assert stopping_distance_m(100.0, 16.0, end_speed_mps=300.0) == 0.0
        assert required_decel_ms2(100.0, 50.0, 300.0) == 0.0

    def test_max_arrival_speed_spends_the_flip_and_the_brake(self):
        a, flip, v_end, d = 15.7, 29.0, 300.0, 350_000.0
        v = max_arrival_speed_mps(d, a, flip, v_end)
        # Coasting through the flip, then braking at a, lands on v_end at d.
        assert v * flip + stopping_distance_m(v, a, v_end) == pytest.approx(d)
        # No room, or no deceleration: only the end speed is allowed.
        assert max_arrival_speed_mps(-5.0, a, flip, v_end) == pytest.approx(v_end)
        assert max_arrival_speed_mps(d, 0.0, flip, v_end) == v_end


# =============================================================================
# INTERCEPT arrival control (R3)
# =============================================================================

class TestInterceptStandoff:

    def test_plain_intercept_still_burns_through_the_target(self):
        """The tool descriptions say plain INTERCEPT never brakes: pin it."""
        sim, me, foe = _duel()
        best_r, best_v, _ = _fly(sim, me, foe, _intercept(), 400)
        assert best_r < 5_000.0
        assert best_v > 2_000.0

    @pytest.mark.parametrize("renew", [False, True])
    def test_standoff_stops_outside_the_standoff_range(self, renew):
        """Two destroyers 400 km apart, INTERCEPT 1.0 standoff 50 km: the
        closest approach stays outside 45 km and the relative speed there is
        within the arrival speed. Renewing the order every 30 s (as every
        captain does) must not reset a ship that has flipped to brake."""
        sim, me, foe = _duel()
        best_r, best_v, log = _fly(sim, me, foe, _intercept(50.0, 0.3, renew=renew), 500,
                                   renew_s=30 if renew else None)
        assert best_r >= 45_000.0
        assert best_v <= 300.0
        modes = [m for _, _, _, m in log]
        assert modes[0] == "CLOSE" and "ARRIVE" in modes and modes[-1] == "HOLD"
        # The flip happens once: CLOSE -> ARRIVE -> HOLD, never back.
        changes = [m for i, m in enumerate(modes) if i == 0 or modes[i - 1] != m]
        assert changes == ["CLOSE", "ARRIVE", "HOLD"]
        # ...and it got there: holding at the standoff range, matched.
        _, r_end, v_end, _ = log[-1]
        assert abs(r_end - 50_000.0) < 3_000.0
        assert v_end < 20.0

    def test_standoff_without_arrival_speed_stops_dead(self):
        sim, me, foe = _duel(stype="corvette")
        best_r, best_v, _ = _fly(sim, me, foe, _intercept(standoff_km=80.0), 400)
        assert best_r >= 75_000.0
        assert best_v < 50.0

    def test_sideways_drift_is_killed_not_orbited(self):
        """Burning straight at the target while drifting sideways settles into
        an orbit where thrust equals v^2/r; the arrival guidance nulls the
        drift and still stops at the standoff."""
        sim, me, foe = _duel(my_vel=(0.0, 1.5, 0.0))
        best_r, best_v, log = _fly(sim, me, foe, _intercept(50.0), 500)
        assert best_r >= 45_000.0
        _, r_end, v_end, _ = log[-1]
        assert abs(r_end - 50_000.0) < 3_000.0
        assert v_end < 20.0

    def test_receding_target_is_chased_then_held(self):
        sim, me, foe = _duel(foe_vel=(1.5, 0.0, 0.0))
        best_r, best_v, log = _fly(sim, me, foe, _intercept(50.0), 900)
        assert best_r >= 45_000.0
        _, r_end, v_end, _ = log[-1]
        assert abs(r_end - 50_000.0) < 3_000.0
        assert v_end < 20.0

    def test_low_throttle_plans_the_stop_on_its_own_thrust(self):
        """The ordered throttle caps every phase, braking included, and the
        stop is planned on that reduced deceleration."""
        sim, me, foe = _duel()
        sim.inject_command("me", _intercept(50.0, throttle=0.4)(sim))
        applied, best_r = [], float("inf")
        for _ in range(900):
            sim.step()
            applied.append(me.applied_throttle)
            best_r = min(best_r, _range_m(me, foe))
        assert best_r >= 45_000.0
        assert max(applied) <= 0.4 + 1e-9
        assert me.execution_mode == "HOLD"

    def test_two_ships_closing_to_one_standoff_meet_at_it(self):
        """Both sides INTERCEPT each other with the same standoff: each
        accounts for the other's charge and braking, no merge, no crawl."""
        sim, me, foe = _duel()
        sim.inject_command("foe", Maneuver(ManeuverType.INTERCEPT, start_time=0.0,
                                           target_id="me", standoff_m=120_000.0))
        best_r, _, log = _fly(sim, me, foe, _intercept(120.0), 600)
        assert best_r >= 115_000.0
        _, r_end, v_end, _ = log[-1]
        assert abs(r_end - 120_000.0) < 3_000.0 and v_end < 20.0
        assert me.execution_mode == foe.execution_mode == "HOLD"

    def test_arrival_speed_alone_arrives_matched_on_the_target(self):
        """Without a standoff there is no room past the target to slow down
        in: the arrival speed is capped to nothing and the ship arrives
        matched on it (it used to pass through it every couple of minutes)."""
        sim, me, foe = _duel()
        _, _, log = _fly(sim, me, foe, _intercept(arrival_kps=1.0), 600, renew_s=30)
        # Braking all the way in (a 1 km/s arrival would cross at 1 km/s).
        near = [v for _, r, v, _ in log if r < 1_000.0]
        assert near and max(near) <= 250.0
        assert log[-1][3] == "HOLD" and log[-1][2] < 20.0
        # Arrived once, then stayed: no periodic pass-throughs.
        far_again = [t for t, r, _, _ in log if t > log[0][0] + 400 and r > 10_000.0]
        assert not far_again

    @pytest.mark.parametrize("standoff_km,arrival_kps", [(50.0, 2.0), (100.0, 3.0)])
    def test_fast_arrival_speed_never_flies_through(self, standoff_km, arrival_kps):
        """An arrival speed is capped to what the overshoot can still be
        stopped from well inside the standoff; once held, re-approaches
        arrive matched (the reviewer's repro passed the target at 0.9 km)."""
        sim, me, foe = _duel()
        best_r, _, log = _fly(sim, me, foe, _intercept(standoff_km, arrival_kps, renew=True),
                              600, renew_s=30)
        assert best_r >= 0.5 * standoff_km * 1000.0
        _, r_end, v_end, mode = log[-1]
        assert abs(r_end - standoff_km * 1000.0) < 3_000.0 and v_end < 20.0
        assert mode == "HOLD" and me.current_maneuver._range_arrived
        # It still kept some momentum: crossed the standoff faster than a
        # matched arrival would have.
        crossing = [v for _, r, v, _ in log if abs(r - standoff_km * 1000.0) < 2_000.0]
        assert max(crossing) > 300.0

    def test_arrival_speed_is_spent_after_the_first_arrival(self):
        sim, me, foe = _duel()
        make = _intercept(50.0, 1.0)
        m = make(sim)
        m._range_arrived = True
        sim.inject_command("me", m)
        sim.step()
        # v_end 0: the cap is the matched-arrival one.
        again = make(sim)
        sim.inject_command("me", again)
        assert again._range_arrived     # carried over on renewal

    def test_no_target_coasts(self):
        sim, me, foe = _duel()
        sim.inject_command("me", Maneuver(ManeuverType.INTERCEPT, start_time=0.0, throttle=1.0,
                                          target_id="ghost", standoff_m=50_000.0))
        sim.step()
        assert me.applied_throttle == 0.0

    def test_renewal_carries_the_phase_only_for_the_same_target(self):
        sim, me, foe = _duel()
        _ship(sim, "foe2", "destroyer", "beta", (0, 400, 0))
        first = _intercept(50.0)(sim)
        sim.inject_command("me", first)
        first._range_phase = "ARRIVE"
        again = _intercept(50.0)(sim)
        sim.inject_command("me", again)
        assert again._range_phase == "ARRIVE"
        other = Maneuver(ManeuverType.INTERCEPT, start_time=0.0, target_id="foe2",
                         standoff_m=50_000.0)
        sim.inject_command("me", other)
        assert other._range_phase is None
        # toward_point / toward_ship carry theirs for the same aim only.
        point = _heading(sim, point_km={"x": 9, "y": 0, "z": 0})
        sim.inject_command("me", point)
        point._range_phase, point._range_side = "ARRIVE", 1.0
        same = _heading(sim, point_km={"x": 9, "y": 0, "z": 0})
        sim.inject_command("me", same)
        assert (same._range_phase, same._range_side) == ("ARRIVE", 1.0)
        moved = _heading(sim, point_km={"x": 9, "y": 5, "z": 0})
        sim.inject_command("me", moved)
        assert moved._range_phase is None


# =============================================================================
# HOLD_RANGE
# =============================================================================

class TestHoldRange:

    def test_holds_range_against_an_incoming_target(self):
        sim, me, foe = _duel(foe_vel=(-1.0, 0.0, 0.0))
        best_r, _, log = _fly(sim, me, foe, _hold(200.0), 600)
        assert best_r >= 195_000.0
        tail = log[-120:]
        assert all(abs(r - 200_000.0) < 3_000.0 for _, r, _, _ in tail)
        assert all(v < 20.0 for _, _, v, _ in tail)

    def test_holds_against_a_charging_target(self):
        """He burns at us at 1 g; a 2 g destroyer holding 200 km matches his
        burn instead of being run down (the old cancel-closure heading
        assumed he coasts)."""
        sim, me, foe = _duel()
        foe.current_maneuver = Maneuver(ManeuverType.INTERCEPT, start_time=0.0, throttle=0.5,
                                        target_id="me")
        best_r, _, log = _fly(sim, me, foe, _hold(200.0), 450)
        assert best_r >= 195_000.0
        tail = log[-100:]
        assert all(abs(r - 200_000.0) < 3_000.0 for _, r, _, _ in tail)
        assert all(v < 20.0 for _, _, v, _ in tail)
        assert me.applied_throttle == pytest.approx(0.5, abs=0.05)

    @pytest.mark.parametrize("make", [_hold(150.0), _intercept(150.0)])
    def test_holds_against_a_charger_at_three_quarters_of_our_acceleration(self, make):
        """A 1.5 g cruiser charges a 2 g destroyer on plain INTERCEPT: the
        relative braking left is a quarter of ours, still enough. The stop is
        planned on it (and on the closure his charge adds during our flip),
        so the destroyer arrives at 150 km instead of flying through him."""
        sim = CombatSimulation(seed=7, time_step=1.0)
        me = _ship(sim, "me", "destroyer", "alpha", (0, 0, 0))
        foe = _ship(sim, "foe", "cruiser", "beta", (400, 0, 0), fwd=(-1, 0, 0))
        foe.current_maneuver = Maneuver(ManeuverType.INTERCEPT, start_time=0.0, throttle=1.0,
                                        target_id="me")
        best_r, _, log = _fly(sim, me, foe, make, 600)
        assert best_r >= 145_000.0
        tail = log[-100:]
        assert all(abs(r - 150_000.0) < 3_000.0 and v < 20.0 for _, r, v, _ in tail)

    def test_equal_charger_is_matched_not_closed_on(self):
        """He out-burns any stop we could plan (equal acceleration): never
        CLOSE on him - match his burn (HOLD) and buy time. The closure his
        head start built stays bounded; nothing flies through."""
        sim, me, foe = _duel()
        foe.current_maneuver = Maneuver(ManeuverType.INTERCEPT, start_time=0.0, throttle=1.0,
                                        target_id="me")
        best_r, _, log = _fly(sim, me, foe, _hold(150.0), 500)
        assert best_r > 100_000.0
        modes = [m for _, _, _, m in log[2:]]
        assert set(modes) <= {"HOLD", "OPEN"}
        closing = []
        for _, r, v, _ in log:
            closing.append(v)
        assert max(closing[60:]) < 700.0
        # ...and it is burning AWAY from him, not toward him.
        los = (foe.position - me.position).normalized()
        assert me.forward.dot(los) < -0.9 and me.applied_throttle > 0.9

    def test_overshoot_while_braking_is_not_read_as_arrived(self):
        """ARRIVE latched outside the wanted range, now inside it and still
        closing at 1.5 km/s: it keeps braking (ARRIVE), it does not call that
        HOLD (the sign-flipped 'toward' used to read as arrived)."""
        sim, me, foe = _duel(foe_km=140.0, my_vel=(1.5, 0.0, 0.0))
        m = _hold(150.0)(sim)
        m._range_phase, m._range_side = "ARRIVE", 1.0
        me.kinematic_state.forward = Vector3D(-1, 0, 0)
        sim.inject_command("me", m)
        sim.step()
        assert m._range_phase == "ARRIVE" and me.execution_mode == "ARRIVE"
        assert me.applied_throttle == pytest.approx(1.0)

    def test_opens_to_the_range_when_too_close(self):
        sim, me, foe = _duel(foe_km=50.0)
        _, _, log = _fly(sim, me, foe, _hold(200.0), 600)
        _, r_end, v_end, mode = log[-1]
        assert abs(r_end - 200_000.0) < 3_000.0
        assert v_end < 20.0 and mode == "HOLD"

    def test_closes_to_the_range_when_too_far(self):
        sim, me, foe = _duel(foe_km=600.0)
        best_r, _, log = _fly(sim, me, foe, _hold(200.0), 600)
        assert best_r >= 195_000.0
        assert abs(log[-1][1] - 200_000.0) < 3_000.0

    def test_default_range_is_cancel_closure_where_you_are(self):
        """No range: hold the range at the order (it latches once, and a
        renewed order keeps the latched range instead of re-latching)."""
        sim, me, foe = _duel(foe_vel=(-0.5, 0.0, 0.0))
        maneuvers = []

        def make(s):
            m = _hold(None, renew=True)(s)
            maneuvers.append(m)
            return m
        _, _, log = _fly(sim, me, foe, make, 600, renew_s=30)
        assert maneuvers[0].hold_range_m == pytest.approx(400_000.0, abs=1_000.0)
        assert all(m.hold_range_m == maneuvers[0].hold_range_m for m in maneuvers)
        _, r_end, v_end, _ = log[-1]
        assert abs(r_end - maneuvers[0].hold_range_m) < 3_000.0
        assert v_end < 20.0

    def test_crossing_target_is_matched(self):
        sim, me, foe = _duel(foe_vel=(-0.5, 1.0, 0.0))
        _, _, log = _fly(sim, me, foe, _hold(300.0, renew=True), 600, renew_s=30)
        tail = log[-100:]
        assert all(abs(r - 300_000.0) < 3_000.0 for _, r, _, _ in tail)
        assert all(v < 20.0 for _, _, v, _ in tail)


# =============================================================================
# HEADING toward a point / ship (CP-03)
# =============================================================================

def _heading(sim, point_km=None, ship_id=None, throttle=1.0):
    return Maneuver(ManeuverType.HEADING, start_time=sim.current_time, duration=0.0,
                    throttle=throttle, heading_point_km=point_km, heading_target_id=ship_id)


class TestHeadingTowardPoint:

    def test_flies_to_the_point_not_along_the_position_vector(self):
        """The Dart-6 case: ship at (187,51,22) km, ordered to (90,55,10) km.
        As a DIRECTION (90,55,10) points 144 deg away from the point; as
        toward_point it reaches the point region."""
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (187, 51, 22))
        foe = _ship(sim, "foe", "destroyer", "beta", (600, 0, 0))
        _coast(foe)
        point = Vector3D(90_000.0, 55_000.0, 10_000.0)
        sim.inject_command("me", _heading(sim, point_km={"x": 90, "y": 55, "z": 10}))
        sim.step()
        h = me.current_maneuver.heading_direction
        bearing = (point - me.position).normalized()
        assert Vector3D(h["x"], h["y"], h["z"]).dot(bearing) > 0.999
        assert bearing.dot(Vector3D(90, 55, 10).normalized()) < -0.75   # the old trap
        closest = float("inf")
        for _ in range(300):
            sim.step()
            closest = min(closest, (me.position - point).magnitude)
        assert closest < 5_000.0

    def test_holds_inside_the_arrival_radius(self):
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0))
        sim.inject_command("me", _heading(sim, point_km={"x": 1.0, "y": 0.0, "z": 0.0}))
        sim.step()
        assert 1_000.0 < HEADING_POINT_ARRIVAL_M
        assert me.applied_throttle == 0.0 and me.execution_mode == "HOLD"

    @pytest.mark.parametrize("stype,side_kps,dist_km,seconds", [
        ("destroyer", 0.0, 200.0, 400),
        ("destroyer", 3.0, 200.0, 650),
        ("battleship", 2.0, 200.0, 850),
        ("corvette", 5.0, 300.0, 650),
    ])
    def test_reaches_and_stops_at_the_point(self, stype, side_kps, dist_km, seconds):
        """Mid-battle ships carry km/s of sideways velocity: burning along
        the bearing only circled the point (the reviewer's repro never got
        closer than it started). The burn now follows the velocity change
        that heads for the point - sideways drift cancelled - and it flips
        and brakes in time: it ends ON the point, stopped, and stays."""
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", stype, "alpha", (0, 0, 0), vel_kps=(0.0, side_kps, 0.0))
        point = Vector3D(dist_km * 1000.0, 0.0, 0.0)
        make = lambda s: Maneuver(ManeuverType.HEADING, start_time=s.current_time, duration=30.0,
                                  throttle=1.0, heading_point_km={"x": dist_km, "y": 0, "z": 0})
        sim.inject_command("me", make(sim))
        arrived = None
        for k in range(seconds):
            if k and k % 30 == 0:
                sim.inject_command("me", make(sim))      # renewed like a captain would
            sim.step()
            d = (me.position - point).magnitude
            if arrived is None and d < HEADING_POINT_ARRIVAL_M and me.velocity.magnitude < 50.0:
                arrived = sim.current_time
            if arrived is not None:
                assert d < 2.5 * HEADING_POINT_ARRIVAL_M      # no fly-through after arrival
        assert arrived is not None
        assert (me.position - point).magnitude < HEADING_POINT_ARRIVAL_M
        assert me.velocity.magnitude < 20.0


class TestHeadingTowardShip:

    def test_nose_tracks_a_crossing_ship(self):
        """Attitude only (throttle 0): the nose follows the target's CURRENT
        bearing as it crosses - a fixed direction would fall behind."""
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0))
        foe = _ship(sim, "foe", "corvette", "beta", (200, -60, 0), vel_kps=(0.0, 2.0, 0.0))
        _coast(foe)
        sim.inject_command("me", _heading(sim, ship_id="foe", throttle=0.0))
        errors = []
        for k in range(120):
            sim.step()
            if k > 60:
                bearing = (foe.position - me.position).normalized()
                errors.append(math.degrees(me.forward.angle_to(bearing)))
        assert max(errors) < 3.0
        assert me.velocity.magnitude == pytest.approx(0.0, abs=1e-6)
        # The bearing swung ~60 deg while we watched.
        start = Vector3D(200, -60, 0).normalized()
        assert math.degrees(start.angle_to(me.forward)) > 40.0

    def test_burns_toward_a_moving_ship(self):
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0))
        foe = _ship(sim, "foe", "destroyer", "beta", (300, 0, 0), vel_kps=(0.0, 1.0, 0.0))
        _coast(foe)
        sim.inject_command("me", _heading(sim, ship_id="foe"))
        start = _range_m(me, foe)
        for _ in range(150):
            sim.step()
        assert _range_m(me, foe) < 0.5 * start
        assert me.current_maneuver.heading_direction is not None

    @pytest.mark.parametrize("my_vel,foe_vel", [((0.0, 3.0, 0.0), (0.0, 0.0, 0.0)),
                                                ((0.0, 0.0, 0.0), (0.0, 3.0, 0.0))])
    def test_joins_the_ship_despite_relative_sideways_velocity(self, my_vel, foe_vel):
        """Pure pursuit with 3 km/s of relative sideways velocity never got
        closer than it started; toward_ship now flies to the ship and falls
        in alongside it, matching its velocity."""
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", "destroyer", "alpha", (0, 0, 0), vel_kps=my_vel)
        foe = _ship(sim, "foe", "destroyer", "beta", (300, 0, 0), vel_kps=foe_vel)
        _coast(foe)
        sim.inject_command("me", _heading(sim, ship_id="foe"))
        for _ in range(700):
            sim.step()
        assert _range_m(me, foe) < 2 * HEADING_POINT_ARRIVAL_M
        assert (me.velocity - foe.velocity).magnitude < 20.0
        assert me.execution_mode == "HOLD"

    def test_dead_ship_coasts(self):
        sim = CombatSimulation(seed=3, time_step=1.0)
        me = _ship(sim, "me", "corvette", "alpha", (0, 0, 0))
        foe = _ship(sim, "foe", "destroyer", "beta", (300, 0, 0))
        _coast(foe)
        foe.is_destroyed = True
        sim.inject_command("me", _heading(sim, ship_id="foe"))
        sim.step()
        assert me.applied_throttle == 0.0


# =============================================================================
# Evasion overlay (R4)
# =============================================================================

def _flank_torpedo_run(evade, seconds=400):
    """A destroyer flying HEADING +x while a corvette 400 km off its flank
    launches one torpedo at it."""
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=42)
    me = _ship(sim, "tgt", "destroyer", "alpha", (0, 0, 0))
    atk = _ship(sim, "atk", "corvette", "beta", (0, 400, 0), fwd=(0, -1, 0))
    sim.inject_command("tgt", Maneuver(ManeuverType.HEADING, start_time=0.0, throttle=0.5,
                                       heading_direction={"x": 1, "y": 0, "z": 0}, evade=evade))
    sim.inject_command("atk", Maneuver(ManeuverType.MAINTAIN, start_time=0.0))
    assert sim.inject_command("atk", {"type": "launch_torpedo", "target_id": "tgt"})
    modes = []
    for _ in range(seconds):
        sim.step()
        modes.append(me.execution_mode)
        if not sim.torpedoes:
            break
    impacts = sim.get_events_by_type(SimulationEventType.TORPEDO_IMPACT)
    return sim, me, modes, impacts


class TestEvasionOverlay:

    def test_reacts_to_threats_predicate(self):
        assert maneuver_reacts_to_threats(Maneuver(ManeuverType.EVASIVE, 0.0))
        assert maneuver_reacts_to_threats(Maneuver(ManeuverType.INTERCEPT, 0.0, evade="dodge"))
        assert not maneuver_reacts_to_threats(Maneuver(ManeuverType.INTERCEPT, 0.0))
        assert not maneuver_reacts_to_threats(None)

    def test_no_threat_flies_the_base_maneuver_exactly(self):
        runs = []
        for evade in (None, "full"):
            sim, me, foe = _duel()
            _fly(sim, me, foe, _intercept(evade=evade), 120)
            runs.append((me.position.to_tuple(), me.velocity.to_tuple(), me.forward.to_tuple()))
        assert runs[0] == runs[1]

    def test_full_runs_and_presents_dodge_never_runs(self):
        _, _, plain_modes, plain = _flank_torpedo_run(None)
        _, _, full_modes, full = _flank_torpedo_run("full")
        _, me, dodge_modes, dodge = _flank_torpedo_run("dodge")
        assert not set(plain_modes) & {"RUN", "PRESENT"}
        assert "RUN" in full_modes and "PRESENT" in full_modes
        assert "RUN" not in dodge_modes and "PRESENT" in dodge_modes
        # Before the terminal phase the dodge ship keeps flying its heading.
        assert dodge_modes[0] == "HEADING"
        # The flank shot hits the passive ship's side; presenting turns the
        # thick nose onto it.
        assert plain and plain[0].data["hit_location"] == "lateral"
        assert dodge and dodge[0].data["hit_location"] == "nose"
        assert full and full[0].data["hit_location"] == "nose"


# =============================================================================
# Recording compatibility
# =============================================================================

class TestRecordedCommand:

    def test_unset_new_fields_stay_out_of_the_record(self):
        """Old orders serialise exactly as before (no null keys), new ones
        carry their parameters."""
        plain = json_value(Maneuver(ManeuverType.INTERCEPT, 0.0, target_id="b"))
        for key in ("standoff_m", "arrival_closing_mps", "hold_range_m", "heading_point_km",
                    "heading_target_id", "evade", "_range_phase"):
            assert key not in plain
        rich = json_value(Maneuver(ManeuverType.INTERCEPT, 0.0, target_id="b",
                                   standoff_m=5e4, evade="dodge"))
        assert rich["standoff_m"] == 5e4 and rich["evade"] == "dodge"

    def test_maneuver_started_event_logs_new_parameters_only_when_set(self):
        sim, me, foe = _duel()
        sim.inject_command("me", _intercept()(sim))
        sim.inject_command("me", _intercept(50.0)(sim))
        started = sim.get_events_by_type(SimulationEventType.MANEUVER_STARTED)
        assert "standoff_m" not in started[0].data
        assert started[1].data["standoff_m"] == 50_000.0


# =============================================================================
# Prediction shadow: the new maneuvers are predicted exactly and never leak
# =============================================================================

class TestPredictionShadow:

    @pytest.mark.parametrize("maneuver,kwargs", [
        (ManeuverType.INTERCEPT, {"throttle": 1.0, "standoff_m": 60_000.0}),
        (ManeuverType.HOLD_RANGE, {"throttle": 1.0, "hold_range_m": 250_000.0}),
        (ManeuverType.HEADING, {"throttle": 0.7, "heading_target_id": "b"}),
        (ManeuverType.HEADING, {"throttle": 1.0, "heading_point_km": {"x": 50, "y": 200, "z": 0}}),
        (ManeuverType.INTERCEPT, {"throttle": 1.0, "evade": "dodge"}),
    ])
    def test_checkpoint_position_matches_actual(self, maneuver, kwargs):
        from tests.test_trajectory_prediction import make_duel, pos, run_to
        from src.trajectory_prediction import predict_plans
        for start in (3.0, 95.0):
            sim = make_duel(maneuver, **kwargs)
            run_to(sim, start)
            plans = predict_plans(sim, start + 30.0)
            run_to(sim, start + 30.0)
            for sid in ("a", "b"):
                assert math.dist(plans[sid]["path"][-1], pos(sim.ships[sid])) < 1.0, (sid, start)

    def test_overlay_and_range_keeping_prediction_mutates_nothing(self):
        """Same contract as the EVASIVE isolation test, for a standoff
        INTERCEPT with the evasion overlay under torpedo fire (PRESENT scoring
        trials impacts on armor; the phase latch lives on the maneuver)."""
        from tests.test_trajectory_prediction import TestIsolation, make_duel, run_to
        from src.torpedo import GuidanceMode, Torpedo, TorpedoSpecs
        from src.simulation import TorpedoInFlight
        from src.trajectory_prediction import predict_plans
        sim = make_duel(ManeuverType.INTERCEPT, a_type="cruiser", throttle=1.0,
                        standoff_m=80_000.0, evade="full")
        run_to(sim, 2.0)
        a = sim.ships["a"]
        torp = Torpedo(specs=TorpedoSpecs.trident(),
                       position=a.position + Vector3D(60_000, 5_000, 0),
                       velocity=a.velocity + Vector3D(-6_000, 0, 0),
                       target_id="a", guidance_mode=GuidanceMode.COLLISION)
        torp.armed = True
        sim.torpedoes.append(TorpedoInFlight(
            torpedo_id="t1", torpedo=torp, source_ship_id="b", launch_time=sim.current_time))
        run_to(sim, 4.0)
        assert a.execution_mode in ("RUN", "PRESENT")
        fingerprint = TestIsolation()._fingerprint
        before = (fingerprint(sim), a.current_maneuver._range_phase)
        predict_plans(sim, 30.0)
        predict_plans(sim, 60.0)
        assert (fingerprint(sim), a.current_maneuver._range_phase) == before
