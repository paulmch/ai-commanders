"""
Torpedo launch sequence (src/torpedo_launch.py): eject from the hull_mounts
rack / cell, separate on RCS, ignite only when clear - and everything that
used to assume a torch lit at launch.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path

import pytest

from src import torpedo_launch as TL
from src import torpedo_link as L
from src import weapon_arcs
from src.physics import Vector3D
from src.simulation import (CombatSimulation, Maneuver, ManeuverType, SimulationEventType,
                            create_ship_from_fleet_data)

ROOT = Path(__file__).resolve().parent.parent
FLEET = json.loads((ROOT / "data" / "fleet_ships.json").read_text())
MOUNTS = json.loads((ROOT / "data" / "hull_mounts.json").read_text())


def _ship(sim, sid, cls, faction, pos, fwd=(1, 0, 0), vel=(0, 0, 0), pd=False):
    s = create_ship_from_fleet_data(sid, cls, faction, FLEET, position=Vector3D(*pos),
                                    velocity=Vector3D(*vel), forward=Vector3D(*fwd))
    if not pd:
        s.point_defense = []
    sim.add_ship(s)
    return s


def _sim(dt=1.0, seed=3):
    sim = CombatSimulation(time_step=dt, decision_interval=1e9, seed=seed)
    sim.echo = False
    return sim


def _step(sim, n=1):
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            out.extend(sim.step())
    return out


def _until_ignition(sim, tf, max_s=30.0):
    events = []
    t_end = sim.current_time + max_s
    while tf.torpedo.in_launch_sequence and sim.current_time < t_end:
        events.extend(_step(sim))
    return events


class _FreshTube:
    fired_rounds = ()


def _duel(cls="corvette", target_at=(500e3, 0, 0), fwd=(1, 0, 0), vel=(0, 0, 0)):
    sim = _sim()
    sh = _ship(sim, "sh", cls, "alpha", (0, 0, 0), fwd=fwd, vel=vel)
    tg = _ship(sim, "tg", "battlecruiser", "beta", target_at, fwd=(-1, 0, 0))
    return sim, sh, tg


# ---------------------------------------------------------------------------
# Ejection geometry: the rounds of the 3D model's rack / cells
# ---------------------------------------------------------------------------

class TestEjectionGeometry:

    @pytest.mark.parametrize("cls", ["corvette", "cruiser_torpedo"])
    def test_every_launcher_lists_its_magazine_rounds(self, cls):
        tubes = [m for m in MOUNTS["hulls"][cls]["mounts"] if m["kind"] == "launcher"]
        weapons = [w for w in FLEET["ships"][cls]["weapons"] if w["type"] == "torpedo_launcher"]
        assert len(tubes) == len(weapons)
        for m, w in zip(tubes, weapons):
            mag = w.get("magazine", FLEET["weapon_types"]["torpedo_launcher"]["magazine"])
            assert len(m["rounds"]) == mag
            for r in m["rounds"]:
                assert math.isclose(math.hypot(*r["eject"]), 1.0, abs_tol=2e-3)
                assert math.isclose(math.hypot(*r["nose"]), 1.0, abs_tol=2e-3)
                # every round sits outside the hull's own axis, ejected outward
                radial = (0.0, r["pos"][1], r["pos"][2])
                assert sum(a * b for a, b in zip(radial, r["eject"])) > 0

    def test_corvette_rail_rounds_alternate_sides_and_point_forward(self):
        rounds = MOUNTS["hulls"]["corvette"]["mounts"][0]["rounds"]
        sides = [math.copysign(1, r["pos"][1]) for r in rounds]
        assert sides == [1, -1] * 4
        assert all(r["nose"] == [1, 0, 0] for r in rounds)
        assert all(abs(r["eject"][1]) > 0.95 for r in rounds)   # outboard

    def test_cruiser_cells_eject_along_their_bank(self):
        tubes = [m for m in MOUNTS["hulls"]["cruiser_torpedo"]["mounts"]
                 if m["kind"] == "launcher"]
        for m in tubes:
            for r in m["rounds"]:
                assert r["eject"] == m["axis"] == r["nose"]

    @pytest.mark.parametrize("cls,launcher", [("corvette", 0), ("cruiser_torpedo", 0),
                                              ("cruiser_torpedo", 2)])
    def test_round_starts_on_its_rack_with_ship_motion_plus_eject(self, cls, launcher):
        sim = _sim()
        # a rolled, tilted hull moving at 3 km/s: body -> world must hold
        fwd = Vector3D(0.6, 0.8, 0.0)
        sh = _ship(sim, "sh", cls, "alpha", (1000, 2000, 3000), fwd=fwd.to_tuple(),
                   vel=(3000, -500, 10))
        sim._arcs_for(sh).roll = 0.4
        _ship(sim, "tg", "battlecruiser", "beta", (500e3, 0, 0))
        for _ in range(launcher):     # tubes before this one are busy
            sh.ready_torpedo_launchers[_].last_launch_time = 0.0
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        torp = sim.torpedoes[-1].torpedo
        mount = [m for m in sim.hull_mounts[cls].mounts if m.kind == "launcher"][launcher]
        k = torp.launch_seq.round_index
        assert k == TL.pick_round(sim, sh, launcher, _FreshTube(), sim.get_ship("tg"))
        pos_b, eject_b, nose_b = mount.rounds[k]
        frame = TL.body_frame(sh)
        want = sh.position + Vector3D(*frame.to_world(*pos_b))
        assert (torp.position - want).magnitude < 1e-6
        dv = torp.velocity - sh.velocity
        assert dv.magnitude == pytest.approx(TL.LAUNCH_EJECT_SPEED_MPS, rel=1e-9)
        assert dv.normalized().dot(Vector3D(*frame.to_world(*eject_b))) > 0.9999
        seq = torp.launch_seq
        assert seq.nose.dot(Vector3D(*frame.to_world(*nose_b))) > 0.9999
        assert seq.mount == mount.slot
        ev = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_LAUNCHED][-1]
        assert ev.data["mount"] == mount.slot and ev.data["round"] == k
        assert ev.data["eject_pos_m"] == [round(c, 2) for c in pos_b]
        assert ev.data["eject_speed_mps"] == TL.LAUNCH_EJECT_SPEED_MPS

    def test_round_nearest_the_target_side_leaves_first(self):
        # target off the port bow: the port rail rounds go first (outer, upper
        # first), then starboard; mirrored, the starboard ones
        for side in (1.0, -1.0):
            sim, sh, tg = _duel(target_at=(300e3, side * 300e3, 0))
            ys = []
            for _ in range(8):
                sh.ready_torpedo_launchers[0].last_launch_time = -1e9
                assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
                ys.append(math.copysign(1, (sim.torpedoes[-1].torpedo.position - sh.position).y))
            assert ys == [side] * 4 + [-side] * 4
            assert sorted(sh.ready_torpedo_launchers[0].fired_rounds) == list(range(8))

    def test_a_friendly_alongside_takes_the_far_side_first(self):
        sim, sh, tg = _duel()
        _ship(sim, "mate", "corvette", "alpha", (0, 2000.0, 0))      # 2 km to port
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        assert (sim.torpedoes[-1].torpedo.position - sh.position).y < 0

    def test_successive_launches_take_the_next_round(self):
        sim, sh, tg = _duel()
        idx = []
        for k in range(3):
            sh.ready_torpedo_launchers[0].last_launch_time = -1e9
            assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
            idx.append(sim.torpedoes[-1].torpedo.launch_seq.round_index)
        assert idx == [0, 1, 2]
        ys = [math.copysign(1, (tf.torpedo.position - sh.position).y) for tf in sim.torpedoes]
        assert ys == [1, -1, 1]            # port, starboard, port: the rails alternate

    def test_spinning_hull_gives_the_round_its_spin(self):
        sim, sh, tg = _duel()
        sh.angular_velocity_rad_s = 0.2
        sh.angular_axis = Vector3D(0, 0, 1)
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        torp = sim.torpedoes[-1].torpedo
        r = torp.position - sh.position
        spin = Vector3D(0, 0, 0.2).cross(r)
        eject = torp.velocity - sh.velocity - spin
        assert eject.magnitude == pytest.approx(TL.LAUNCH_EJECT_SPEED_MPS, rel=1e-9)
        assert (torp.launch_seq.omega - Vector3D(0, 0, 0.2)).magnitude < 1e-12

    def test_no_mount_data_falls_back_to_the_dorsal_surface(self):
        sim, sh, tg = _duel()
        sim.hull_mounts = {}
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        torp = sim.torpedoes[-1].torpedo
        assert (torp.position - sh.position).z == pytest.approx(sh.geometry.radius_m)
        assert torp.launch_seq.mount is None

    def test_switch_off_restores_the_old_launch(self):
        sim, sh, tg = _duel()
        sim.launch_sequence = False
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        torp = sim.torpedoes[-1].torpedo
        assert torp.launch_seq is None and not torp.in_launch_sequence
        assert torp.position == sh.position and torp.velocity == sh.velocity
        _step(sim)
        assert torp.last_main_throttle > 0      # torch from the first tick


# ---------------------------------------------------------------------------
# The ignition rules
# ---------------------------------------------------------------------------

class TestIgnitionRules:

    def test_phases_eject_separate_boost(self):
        sim, sh, tg = _duel()
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        phases = []
        while tf.torpedo.in_launch_sequence:
            _step(sim)
            phases.append((tf.torpedo.launch_seq.phase, tf.torpedo.last_main_throttle,
                           tf.torpedo.last_rcs_thrust.magnitude))
        assert phases[0] == ("eject", 0.0, 0.0)               # no thruster fires
        assert all(p[0] == "separate" and p[1] == 0.0 for p in phases[1:-1])
        assert any(p[2] > 0 for p in phases[1:])              # RCS puffs clear
        _step(sim)
        assert tf.torpedo.launch_seq.phase == "boost" and tf.torpedo.last_main_throttle > 0

    @pytest.mark.parametrize("cls", ["corvette", "cruiser_torpedo"])
    @pytest.mark.parametrize("target", [(500e3, 0, 0), (0, 500e3, 0), (-500e3, 0, 0),
                                        (0, 0, -500e3)])
    def test_never_lights_inside_the_clear_distance_or_on_its_hull(self, cls, target):
        sim, sh, tg = _duel(cls, target)
        n = 4 if cls == "cruiser_torpedo" else 1
        for _ in range(n):
            assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        events = []
        for _ in range(20):
            snap = TL.ShipSnap(sh)
            states = {tf.torpedo_id: (tf.torpedo.position, tf.torpedo.launch_seq.nose)
                      for tf in sim.torpedoes}
            ev = _step(sim)
            for e in ev:
                if e.event_type != SimulationEventType.TORPEDO_IGNITION:
                    continue
                events.append(e)
                p, nose = states[e.data["torpedo_id"]]
                assert TL.hull_gap_m(snap, p) >= TL.LAUNCH_CLEAR_M
                if e.data["reason"] == "clear":
                    h = Vector3D(*e.data["heading"])
                    assert not TL.exhaust_on_hull(snap, p, h)
                    assert TL.angle_between(nose, h) <= math.radians(TL.LAUNCH_ALIGN_DEG) + 1e-9
        assert len(events) == n
        assert all(e.data["delay_s"] <= TL.LAUNCH_TIMEOUT_S for e in events)

    def test_exhaust_cone_holds_the_torch(self):
        snap = TL.ShipSnap(create_ship_from_fleet_data(
            "x", "corvette", "alpha", FLEET, position=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0)))
        p = Vector3D(0, 0, 200.0)                  # 200 m above the hull
        assert TL.exhaust_on_hull(snap, p, Vector3D(0, 0, 1))      # exhaust down at it
        assert not TL.exhaust_on_hull(snap, p, Vector3D(0, 0, -1))  # exhaust up, away
        assert not TL.exhaust_on_hull(snap, p, Vector3D(1, 0, 0))   # square to it

    def test_timeout_lights_once_clear_even_when_not_aligned(self):
        sim, sh, tg = _duel()
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        seq = sim.torpedoes[-1].torpedo.launch_seq
        snaps = TL.snapshot(sim)
        torp = sim.torpedoes[-1].torpedo
        torp.position = sh.position + Vector3D(0, 500, 0)       # well clear
        seq.heading = Vector3D(-1, 0, 0)                       # nose 180 deg off
        assert TL.check_release(seq, torp, snaps, TL.LAUNCH_TIMEOUT_S - 1) is None
        assert TL.check_release(seq, torp, snaps, TL.LAUNCH_TIMEOUT_S) == "timeout"
        torp.position = sh.position + Vector3D(0, 20, 0)        # not clear
        assert TL.check_release(seq, torp, snaps, TL.LAUNCH_TIMEOUT_S) is None
        assert TL.check_release(seq, torp, snaps, TL.LAUNCH_HARD_TIMEOUT_S) == "hard_timeout"
        assert TL.check_release(seq, torp, snaps, 0.5) is None       # still ejecting

    def test_own_ship_destroyed_mid_sequence_frees_the_round(self):
        sim, sh, tg = _duel()
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        _step(sim)                               # eject done, 12 m off the hull
        sh.is_destroyed = True
        ev = _until_ignition(sim, tf)
        ign = [e for e in ev if e.event_type == SimulationEventType.TORPEDO_IGNITION]
        assert ign and ign[0].data["reason"] == "free" and ign[0].data["own_gap_m"] is None
        assert ign[0].data["delay_s"] < 4.0     # no hull to clear: as soon as it is aligned
        _step(sim)
        assert tf in sim.torpedoes and tf.torpedo.last_main_throttle > 0

    def test_salvo_rounds_clear_each_other(self):
        sim, sh, tg = _duel("cruiser_torpedo")
        for _ in range(4):
            assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        slots = [tf.torpedo.launch_seq.mount for tf in sim.torpedoes]
        assert slots == ["weapon_0", "weapon_1", "weapon_2", "weapon_3"]
        closest = math.inf
        for _ in range(12):
            _step(sim)
            ps = [tf.torpedo.position for tf in sim.torpedoes]
            closest = min(closest, min(ps[i].distance_to(ps[j])
                                       for i in range(4) for j in range(i + 1, 4)))
        assert closest > 40.0

    def test_friendly_hull_holds_the_torch_and_is_never_struck(self):
        sim = _sim()
        sh = _ship(sim, "sh", "cruiser_torpedo", "alpha", (0, 0, 0))
        mate = _ship(sim, "mate", "cruiser_torpedo", "alpha", (0, 120.0, 0))
        _ship(sim, "tg", "battlecruiser", "beta", (500e3, 0, 0), fwd=(-1, 0, 0))
        for s in (sh, mate):
            for _ in range(4):
                assert sim._handle_torpedo_command(s, {"target_id": "tg"})
        worst = math.inf
        for _ in range(20):
            snaps = TL.snapshot(sim)
            ev = _step(sim)
            for tf in sim.torpedoes:
                other = snaps["mate" if tf.source_ship_id == "sh" else "sh"]
                worst = min(worst, TL.hull_gap_m(other, tf.torpedo.position))
            for e in ev:
                assert e.event_type != SimulationEventType.TORPEDO_IMPACT
        assert worst > 10.0
        assert len(sim.torpedoes) == 8

    def test_turning_and_burning_hull_never_sweeps_through_its_round(self):
        for cls in ("corvette", "cruiser_torpedo"):
            sim, sh, tg = _duel(cls)
            c, s = math.cos(math.radians(120)), math.sin(math.radians(120))
            sh.current_maneuver = Maneuver(maneuver_type=ManeuverType.HEADING, start_time=0.0,
                                           duration=1e9, throttle=1.0,
                                           heading_direction={"x": c, "y": s, "z": 0.0})
            _step(sim, 3)
            assert sh.angular_velocity_rad_s > 0.02          # mid-turn at launch
            n = 4 if cls == "cruiser_torpedo" else 1
            for _ in range(n):
                assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
            for _ in range(15):
                _step(sim)
                for tf in sim.torpedoes:
                    assert TL.hull_gap_m(TL.ShipSnap(sh), tf.torpedo.position) > 5.0

    def test_corvette_burning_away_keeps_its_round_out_of_its_plume(self):
        from src.drive_plume import PLUME_MAX_HALF_ANGLE_DEG
        sim, sh, tg = _duel("corvette", (-500e3, 0, 0))
        sh.current_maneuver = Maneuver(maneuver_type=ManeuverType.HEADING, start_time=0.0,
                                       duration=1e9, throttle=1.0,
                                       heading_direction={"x": 1.0, "y": 0.0, "z": 0.0})
        _step(sim, 3)
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        while tf.torpedo.in_launch_sequence:
            _step(sim)
            tail = sh.position - sh.forward * (sh.geometry.length_m / 2)
            ang = math.degrees(TL.angle_between(sh.forward * -1.0, tf.torpedo.position - tail))
            assert ang > PLUME_MAX_HALF_ANGLE_DEG

    def test_seeker_estimate_sees_the_target_not_the_ejection_kick(self):
        # coasting target: the estimate stays zero through eject, separation
        # and ignition; a target burning at 1 g is measured as 1 g by then
        for burn in (False, True):
            sim, sh, tg = _duel("cruiser_torpedo", target_at=(400e3, 0, 0))
            if burn:
                tg.current_maneuver = Maneuver(maneuver_type=ManeuverType.HEADING,
                                               start_time=0.0, duration=1e9, throttle=1.0,
                                               heading_direction={"x": 0.0, "y": 1.0, "z": 0.0})
                _step(sim, 30)                     # turned onto the burn
            assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
            tf = sim.torpedoes[-1]
            _until_ignition(sim, tf)
            _step(sim)
            est = tf.torpedo.estimated_target_accel.magnitude
            if burn:
                assert est == pytest.approx(tg.kinematic_state.max_acceleration_ms2(), rel=0.1)
            else:
                assert est < 1e-9

    def test_ignition_cancels_the_separation_drift(self):
        sim, sh, tg = _duel("cruiser_torpedo")
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        _until_ignition(sim, tf)
        los = (tg.position - tf.torpedo.position).normalized()
        drift = tf.torpedo.velocity - sh.velocity
        lateral = drift - los * drift.dot(los)
        assert lateral.magnitude < 1.0                        # m/s


# ---------------------------------------------------------------------------
# Determinism, mirror symmetry, prediction
# ---------------------------------------------------------------------------

def _mirror_run(order, mirror_y=False):
    sim = _sim(seed=11)
    sgn = -1.0 if mirror_y else 1.0
    a = dict(sid="a", cls="cruiser_torpedo", faction="alpha", pos=(-250e3, 0, 0), fwd=(1, 0, 0))
    b = dict(sid="b", cls="cruiser_torpedo", faction="beta", pos=(250e3, 0, 0), fwd=(-1, 0, 0))
    for spec in (order if order else (a, b)):
        _ship(sim, spec["sid"], spec["cls"], spec["faction"], spec["pos"], fwd=spec["fwd"])
    for _ in range(2):
        assert sim._handle_torpedo_command(sim.get_ship("a"), {"target_id": "b"})
        assert sim._handle_torpedo_command(sim.get_ship("b"), {"target_id": "a"})
    trace = []
    for _ in range(12):
        _step(sim)
        trace.append({tf.torpedo_id: (tf.torpedo.position.to_tuple(),
                                      tf.torpedo.launch_seq.phase) for tf in sim.torpedoes})
    return trace, (a, b)


class TestSymmetryAndDeterminism:

    def test_mirror_sides_fly_mirror_sequences(self):
        trace, _ = _mirror_run(None)
        for frame in trace:
            for k in range(1, 3):
                pa, pha = frame[f"torp_a_{2 * k - 1:04d}"]
                pb, phb = frame[f"torp_b_{2 * k:04d}"]
                assert pha == phb
                # b's rounds: x mirrored, its port (+y body) is world -y
                assert pa[0] == pytest.approx(-pb[0], abs=1e-6)
                assert pa[1] == pytest.approx(-pb[1], abs=1e-6)
                assert pa[2] == pytest.approx(pb[2], abs=1e-6)

    def test_insertion_order_does_not_matter(self):
        t1, (a, b) = _mirror_run(None)
        t2, _ = _mirror_run((b, a))
        assert t1 == t2

    def test_deterministic(self):
        assert _mirror_run(None)[0] == _mirror_run(None)[0]

    def test_prediction_shadow_never_touches_the_real_sequence(self):
        from src.trajectory_prediction import PredictionSimulation
        sim, sh, tg = _duel()
        tg.current_maneuver = Maneuver(maneuver_type=ManeuverType.EVASIVE, start_time=0.0,
                                       duration=1e9, throttle=1.0)
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        _step(sim)
        before = (tf.torpedo.position.to_tuple(), tf.torpedo.launch_seq.phase,
                  tf.torpedo.launch_seq.nose.to_tuple())
        shadow = PredictionSimulation(sim)
        for _ in range(10):
            shadow.advance(1.0)
        assert (tf.torpedo.position.to_tuple(), tf.torpedo.launch_seq.phase,
                tf.torpedo.launch_seq.nose.to_tuple()) == before
        assert tf.torpedo.in_launch_sequence


# ---------------------------------------------------------------------------
# Nobody treats a separating round as an inbound threat
# ---------------------------------------------------------------------------

class TestNotAThreatYet:

    def test_pd_does_not_engage_a_separating_round(self):
        sim = _sim()
        sh = _ship(sim, "sh", "corvette", "alpha", (0, 0, 0))
        tg = _ship(sim, "tg", "battlecruiser", "beta", (30e3, 0, 0), fwd=(-1, 0, 0), pd=True)
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        pd_events = []
        while tf.torpedo.in_launch_sequence:
            pd_events += [e for e in _step(sim) if e.event_type.name.startswith("PD_")]
        assert not pd_events
        assert tf.heat_absorbed_j == 0.0

    def test_target_does_not_evade_before_ignition(self):
        sim, sh, tg = _duel()
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        assert not sim._torpedo_inbound(tg, 1e9)
        assert sim._get_evasion_status(tg)["guided_torpedoes"] == 0
        _until_ignition(sim, tf)
        assert sim._torpedo_inbound(tg, 1e9)


# ---------------------------------------------------------------------------
# Forecasts include the delay and match the flown time
# ---------------------------------------------------------------------------

class TestForecasts:

    @pytest.mark.parametrize("cls,target", [
        ("corvette", (500e3, 0, 0)), ("corvette", (0, 500e3, 0)), ("corvette", (-500e3, 0, 0)),
        ("cruiser_torpedo", (500e3, 0, 0)), ("cruiser_torpedo", (0, -500e3, 0))])
    def test_hold_forecast_is_the_flown_ignition(self, cls, target):
        sim, sh, tg = _duel(cls, target)
        fc = TL.predict_launch_hold_s(sim, sh, tg)
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        assert tf.torpedo.launch_seq.hold_forecast_s == fc
        ev = _until_ignition(sim, tf)
        flown = [e for e in ev if e.event_type == SimulationEventType.TORPEDO_IGNITION][0]
        assert abs(flown.data["delay_s"] - fc) <= 1.0          # one tick

    def test_hold_left_counts_down(self):
        sim, sh, tg = _duel()
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        torp = sim.torpedoes[-1].torpedo
        h0 = torp.launch_hold_s()
        _step(sim)
        assert torp.launch_hold_s() == pytest.approx(h0 - 1.0)
        _until_ignition(sim, sim.torpedoes[-1])
        assert torp.launch_hold_s() == 0.0

    @pytest.mark.parametrize("rng_km", [50.0, 500.0])
    def test_captain_shot_tof_is_the_flown_one_and_counts_the_delay(self, rng_km):
        from src.llm.fire_solutions import fly_torpedo
        sim, sh, tg = _duel(target_at=(rng_km * 1000, 0, 0))
        shot = fly_torpedo(sim, sh, tg, FLEET, target_evades=False)
        sim.launch_sequence = False
        instant = fly_torpedo(sim, sh, tg, FLEET, target_evades=False)
        sim.launch_sequence = True
        assert shot.hits and instant.hits
        assert shot.tof_s > instant.tof_s
        # the real launch, flown in the battle sim
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        t0 = sim.current_time
        hit_t = None
        for _ in range(200):
            for e in _step(sim):
                if e.event_type == SimulationEventType.TORPEDO_IMPACT:
                    hit_t = e.timestamp - t0
            if hit_t is not None:
                break
        assert hit_t is not None
        assert abs((hit_t + 1.0) - shot.tof_s) <= 1.0

    def test_board_eta_includes_the_hold(self):
        sim, sh, tg = _duel()
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        torp = sim.torpedoes[-1].torpedo
        fc = L.round_forecast(torp, tg)
        seq, torp.launch_seq = torp.launch_seq, None
        fc0 = L.round_forecast(torp, tg)
        torp.launch_seq = seq
        assert fc.eta_s == pytest.approx(fc0.eta_s + seq.hold_forecast_s, abs=0.5)
        assert sim.torpedo_board("alpha")["tg"]["eta"] == [fc.eta_s]

    def test_fleet_net_eta_of_a_separating_round_is_its_flight(self):
        from src.llm.fleet_comms import torpedo_eta
        sim, sh, tg = _duel(target_at=(300e3, 0, 0))
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        eta = torpedo_eta(tf, tg)                  # launcher at rest: not "closing"
        assert eta is not None
        t0, hit_t = sim.current_time, None
        while sim.torpedoes and hit_t is None:
            for e in _step(sim):
                if e.event_type == SimulationEventType.TORPEDO_IMPACT:
                    hit_t = sim.current_time - t0
        assert abs(eta - hit_t) <= 2.0

    def test_admiral_pursuit_counts_the_hold(self):
        from src.llm import fleet_intel as intel
        specs = create_ship_from_fleet_data("x", "corvette", "alpha", FLEET) \
            .ready_torpedo_launchers[0].specs
        a = intel.solve_pursuit(300e3, 0.0, 0.0, 0.0, specs)
        b = intel.solve_pursuit(300e3, 0.0, 0.0, 0.0, specs, hold_s=4.0)
        assert b["tof_s"] == pytest.approx(a["tof_s"] + 4.0, abs=1.0)

    def test_can_intercept_counts_the_hold(self):
        sim, sh, tg = _duel()
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        torp = sim.torpedoes[-1].torpedo
        # an opening target needs the powered intercept: the hold adds to it
        tg.kinematic_state.velocity = Vector3D(2000, 0, 0)
        t_seq = torp._time_to_intercept(tg.position, tg.velocity, Vector3D.zero())
        seq, torp.launch_seq = torp.launch_seq, None
        t0 = torp._time_to_intercept(tg.position, tg.velocity, Vector3D.zero())
        torp.launch_seq = seq
        assert t_seq == pytest.approx(t0 + seq.hold_forecast_s)


# ---------------------------------------------------------------------------
# Tubes, profiles, datalink, cleanup
# ---------------------------------------------------------------------------

class TestRestOfTheGame:

    def test_tube_cycle_starts_at_ejection(self):
        sim, sh, tg = _duel()
        tube = sh.ready_torpedo_launchers[0]
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        assert tube.last_launch_time == sim.current_time
        assert not sim._handle_torpedo_command(sh, {"target_id": "tg"})   # cooling

    def test_wrecked_tube_does_not_take_the_round_already_out(self):
        sim, sh, tg = _duel()
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        sh.ready_torpedo_launchers[0].current_magazine = 0      # a wrecked tube's loss
        _until_ignition(sim, tf)
        assert tf in sim.torpedoes and not tf.torpedo.in_launch_sequence

    @pytest.mark.parametrize("profile", ["sprint", "reserve"])
    def test_both_profiles_start_with_the_same_sequence(self, profile):
        sim, sh, tg = _duel()
        assert sim._handle_torpedo_command(sh, {"target_id": "tg", "profile": profile})
        tf = sim.torpedoes[-1]
        assert tf.torpedo.profile == profile
        ev = _until_ignition(sim, tf)
        ign = [e for e in ev if e.event_type == SimulationEventType.TORPEDO_IGNITION]
        assert ign[0].data["delay_s"] == 4.0

    def test_round_retargeted_while_separating_carries_on(self):
        sim = _sim()
        # closing 3 km/s, so the datalink has a coasting intercept to offer
        sh = _ship(sim, "sh", "corvette", "alpha", (0, 0, 0), vel=(3000, 0, 0))
        _ship(sim, "tg", "battlecruiser", "beta", (300e3, 0, 0), fwd=(-1, 0, 0))
        _ship(sim, "tg2", "battlecruiser", "beta", (320e3, 5e3, 0), fwd=(-1, 0, 0))
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        _step(sim)
        sim.get_ship("tg").is_destroyed = True
        _step(sim)
        assert tf.torpedo.target_id == "tg2"
        assert tf.torpedo.in_launch_sequence or tf.torpedo.launch_seq.ignition_t is not None
        _until_ignition(sim, tf)
        assert not tf.torpedo.in_launch_sequence

    def test_orphan_in_sequence_never_strikes_its_own_ship(self):
        sim = _sim()
        sim.torpedo_datalink = False
        sh = _ship(sim, "sh", "corvette", "alpha", (0, 0, 0))
        tg = _ship(sim, "tg", "battlecruiser", "beta", (300e3, 0, 0), fwd=(-1, 0, 0))
        assert sim._handle_torpedo_command(sh, {"target_id": "tg"})
        tf = sim.torpedoes[-1]
        # point the orphan straight back through its launcher
        tf.torpedo.velocity = sh.velocity + (sh.position - tf.torpedo.position).normalized() * 20
        tf.is_disabled = True
        for _ in range(5):
            for e in _step(sim):
                assert e.event_type != SimulationEventType.TORPEDO_IMPACT
        assert not sh.is_destroyed and sh.hull_integrity == 100


# ---------------------------------------------------------------------------
# Replay fields
# ---------------------------------------------------------------------------

def test_trace_carries_phase_nose_and_rcs(tmp_path):
    from src.llm.battle_recorder import BattleRecorder
    rec = BattleRecorder()
    rec._is_recording = True
    rec.recording.sim_trace = []
    torp = {"id": "t", "position": (0, 0, 0), "velocity": (0, 0, 0), "source_ship_id": "a",
            "target_ship_id": "b", "main_throttle": 0.0, "main_direction": None,
            "rcs_thrust": (0.0, 1.0, 0.0), "launch_phase": "separate",
            "nose": (0.0, 0.70710678, 0.70710678)}
    rec.record_sim_frame(1.0, {}, [], [torp])
    entry = rec.recording.sim_trace[-1]["torpedoes"][0]
    assert entry["ph"] == "separate" and entry["nose"] == [0.0, 0.7071, 0.7071]
    assert entry["rcs"] == [0.0, 1.0, 0.0] and entry["thr"] == 0.0
    # once the torch burns (and on every old recording): no ph / nose
    boost = dict(torp, launch_phase=None, main_throttle=1.0, main_direction=(1, 0, 0))
    boost.pop("nose")
    rec.record_sim_frame(2.0, {}, [], [boost])
    entry = rec.recording.sim_trace[-1]["torpedoes"][0]
    assert "ph" not in entry and "nose" not in entry and entry["fwd"] == [1, 0, 0]
