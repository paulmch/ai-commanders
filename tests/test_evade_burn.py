"""
The evade overlay on a burning order (owner's rule, pass 5; the rule is
written down in the src/simulation.py module docstring): "with evade the
engine burns the whole time using thrust vectoring to turn".

- the drive never runs below what the order burns while evasion has the helm;
- the thrust axis stays inside EVADE_BURN_CONE_DEG of the burn line, so most
  of the push stays along it (cos 20 deg = 94 % at the cone edge);
- each hull's cone and swing period are ones its thrust vectoring can fly;
- the swing draws no battle RNG and repeats exactly for a seed;
- RUN never drops below the order's throttle;
- engine-off orders keep the older overlay (BRACE drive off, TANK half).
"""

import json
import math
from pathlib import Path

import pytest

from src.firecontrol import WeaponsCommand, WeaponsOrder
from src.physics import Vector3D
from src.simulation import (
    EVADE_BURN_CONE_DEG,
    EVADE_BURN_CYCLE_MAX_S,
    EVADE_BURN_CYCLE_MIN_S,
    EVASION_OVERLAY_MODES,
    HEADING_BURN_CONE_DEG,
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    create_ship_from_fleet_data,
)

FLEET = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())
CONE = math.radians(EVADE_BURN_CONE_DEG)
LINE = Vector3D(0.0, 1.0, 0.0)


def _under_fire(hull="destroyer", throttle=1.0, evade="dodge", seed=3, maneuver=None):
    """``hull`` at the origin flying HEADING +y (across the line of sight,
    nose already on it) while a battlecruiser and two destroyers 150 km
    down +x fire every gun at it."""
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    me = create_ship_from_fleet_data("me", hull, "alpha", FLEET, position=Vector3D(0, 0, 0),
                                     velocity=Vector3D(0, 0, 0), forward=LINE)
    sim.add_ship(me)
    for i, st in enumerate(("battlecruiser", "destroyer", "destroyer")):
        pos = Vector3D(150e3, (i - 1) * 25e3, 0.0)
        s = create_ship_from_fleet_data(f"s{i}", st, "beta", FLEET, position=pos,
                                        forward=(me.position - pos).normalized())
        sim.add_ship(s)
        s.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, throttle=0.0,
                                      target_id="me")
        s.weapons_orders = {
            slot: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, slot, "me", min_hit_probability=0.0)
            for slot in s.weapons if not slot.startswith("pd_")}
    me.current_maneuver = maneuver or Maneuver(
        ManeuverType.HEADING, 0, duration=1e9, throttle=throttle,
        heading_direction={"x": 0.0, "y": 1.0, "z": 0.0}, evade=evade)
    return sim, me


def _fly(sim, me, seconds):
    """Per step: (mode, applied throttle, nose angle off the line, swing,
    engine fraction)."""
    rows = []
    for _ in range(seconds):
        sim.step()
        if me.is_destroyed or me.is_dying:
            break
        rows.append((me.execution_mode, me.applied_throttle, me.forward.angle_to(LINE),
                     me.evade_swing, me.get_effective_thrust_fraction()))
    return rows


class TestTheDriveKeepsTheOrder:

    @pytest.mark.parametrize("evade", ["dodge", "full"])
    @pytest.mark.parametrize("throttle", [1.0, 0.6])
    def test_never_below_the_ordered_throttle_while_evading(self, evade, throttle):
        sim, me = _under_fire(throttle=throttle, evade=evade)
        rows = _fly(sim, me, 120)
        evading = [r for r in rows if r[0] in EVASION_OVERLAY_MODES]
        assert len(evading) >= 20, "the overlay must have flown under this fire"
        for mode, applied, off, swing, engine in rows:
            # damage may cut the drive (engine fraction), evasion may not
            assert applied == pytest.approx(throttle * engine, abs=1e-9), (mode, applied, off)
        assert all(r[3] for r in evading)           # every evasion step swung the burn

    def test_the_thrust_axis_stays_in_the_cone(self):
        sim, me = _under_fire()
        rows = _fly(sim, me, 120)
        gate = CONE + math.radians(HEADING_BURN_CONE_DEG)
        assert max(r[2] for r in rows) <= gate
        # and the swing is not degenerate: the nose does leave the line
        assert max(r[2] for r in rows) > math.radians(10.0)

    def test_along_track_share_is_at_least_cos_cone(self):
        sim, me = _under_fire()
        rows = _fly(sim, me, 120)
        under = [r for r in rows if r[0] in EVASION_OVERLAY_MODES]
        share = sum(math.cos(r[2]) for r in under) / len(under)
        assert share >= math.cos(CONE) - 0.01
        assert share < 0.999                        # it does swing

    def test_after_the_threat_the_burn_swings_back_without_idling(self):
        # Between threats the order flies; a nose the swing left off the line
        # swings back at the order's throttle (no turn-first token).
        sim, me = _under_fire()
        rows = _fly(sim, me, 120)
        back = [r for r in rows if r[0] not in EVASION_OVERLAY_MODES
                and r[2] > math.radians(HEADING_BURN_CONE_DEG)]
        assert back, "the scenario must leave the nose off the line after a swing"
        assert all(r[1] == pytest.approx(1.0 * r[4]) for r in back)

    def test_the_old_overlay_held_the_burn_here(self, monkeypatch):
        # The measurement behind the rule: without the swing (the older
        # overlay: EVADE full-throttle jinks across, TANK half, BRACE off)
        # this ship spent most of its time under fire off its order.
        sim, me = _under_fire()
        monkeypatch.setattr(CombatSimulation, "_order_burn",
                            lambda self, *a, **k: None)
        rows = _fly(sim, me, 120)
        under = [r for r in rows if r[0] in EVASION_OVERLAY_MODES]
        on_order = [r for r in under if r[1] >= 0.99 and r[2] <= CONE]
        assert len(on_order) < 0.5 * len(under)


class TestEachHullCanFlyItsSwing:

    def test_cone_and_cycle_come_from_the_hull(self):
        sim = CombatSimulation(time_step=1.0, seed=1)
        seen = {}
        for cls in FLEET["ships"]:
            ship = create_ship_from_fleet_data(cls, cls, "alpha", FLEET, position=Vector3D(0, 0, 0))
            cone, cycle = sim._evade_burn_swing(ship)
            assert 0.0 < cone <= CONE + 1e-12
            assert EVADE_BURN_CYCLE_MIN_S <= cycle <= EVADE_BURN_CYCLE_MAX_S
            # edge to edge from rest on thrust vectoring fits in one cycle
            assert sim._estimate_turn_time(ship, 2.0 * cone, engines_on=True) <= cycle + 1e-6
            seen[cls] = (math.degrees(cone), cycle)
        # a corvette swings the full cone on the jink cycle; a dreadnought
        # cannot cross 40 degrees in 30 s and swings a narrower cone
        assert seen["corvette"] == pytest.approx((EVADE_BURN_CONE_DEG, EVADE_BURN_CYCLE_MIN_S))
        assert seen["dreadnought"][0] < EVADE_BURN_CONE_DEG
        assert seen["dreadnought"][1] == EVADE_BURN_CYCLE_MAX_S
        assert seen["destroyer"][1] < seen["battleship"][1]

    def test_a_damaged_drive_swings_slower(self):
        sim = CombatSimulation(time_step=1.0, seed=1)
        ship = create_ship_from_fleet_data("x", "cruiser", "alpha", FLEET, position=Vector3D(0, 0, 0))
        fresh = sim._evade_burn_swing(ship)
        ship.get_effective_turn_rate_multiplier = lambda: 0.3
        damaged = sim._evade_burn_swing(ship)
        assert damaged[1] > fresh[1] or damaged[0] < fresh[0]

    @pytest.mark.parametrize("hull", ["corvette", "battlecruiser", "dreadnought"])
    def test_the_flown_swing_respects_the_turn_authority(self, hull):
        sim, me = _under_fire(hull=hull)
        accel, max_vel = sim._get_rotation_params(me, True)
        prev = None
        for _ in range(90):
            sim.step()
            if me.is_destroyed or me.is_dying:
                break
            w = me.angular_velocity_rad_s
            assert w <= max_vel + 1e-9
            if prev is not None:
                assert abs(w - prev) <= accel * 1.0 + 1e-9
            prev = w


class TestRandomnessAndDeterminism:

    def test_the_swing_draws_no_battle_rng(self, monkeypatch):
        sim, me = _under_fire()
        calls = []
        original = CombatSimulation._swing_burn

        def wrapped(self, *args, **kwargs):
            before = self.rng.getstate()
            out = original(self, *args, **kwargs)
            calls.append(before == self.rng.getstate())
            return out

        monkeypatch.setattr(CombatSimulation, "_swing_burn", wrapped)
        _fly(sim, me, 90)
        assert calls and all(calls)

    def test_a_seed_repeats_exactly(self):
        runs = []
        for _ in range(2):
            sim, me = _under_fire(seed=11)
            _fly(sim, me, 90)
            runs.append([(e.timestamp, e.event_type.name, e.ship_id,
                          json.dumps(e.data, sort_keys=True, default=str)) for e in sim.events])
        assert runs[0] == runs[1]

    def test_the_side_changes_with_the_swing_cycle(self):
        # Against the same fire the swing leaves the line on both sides over
        # a run: the side is drawn per cycle, not fixed by the geometry.
        sim, me = _under_fire(hull="corvette", seed=5)
        sides = set()
        for _ in range(120):
            sim.step()
            if me.evade_swing and me.forward.angle_to(LINE) > math.radians(10.0):
                sides.add(me.forward.z > 0)
        assert sides == {True, False}


class TestRunAndEngineOffOrders:

    def test_run_never_drops_below_the_orders_throttle(self):
        from tests.test_maneuver_semantics import _flank_torpedo_run
        steps = []
        _, _, modes, _ = _flank_torpedo_run("full", throttle=0.5, applied=steps)
        run = [thr for mode, thr in steps if mode == "RUN"]
        assert run and min(run) >= 0.5 - 1e-9

    def test_padlock_dodge_keeps_the_older_overlay(self):
        # An engine-off order: TANK burns at half throttle and BRACE cuts the
        # drive, as before (nothing to keep burning).
        m = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, throttle=1.0, target_id="s0",
                     evade="dodge")
        sim, me = _under_fire(maneuver=m)
        rows = _fly(sim, me, 120)
        modes = {r[0]: [] for r in rows}
        for r in rows:
            modes[r[0]].append(r)
        assert "BRACE" in modes and all(r[1] == 0.0 for r in modes["BRACE"])
        assert "TANK" in modes and all(r[1] == pytest.approx(0.5 * r[4]) for r in modes["TANK"])
        assert not any(r[3] for r in rows)          # never the burn swing

    def test_a_trimming_range_keeper_is_engine_off_for_the_overlay(self):
        sim = CombatSimulation(time_step=1.0, seed=1)
        ship = create_ship_from_fleet_data("x", "destroyer", "alpha", FLEET, position=Vector3D(0, 0, 0))
        sim.add_ship(ship)
        m = Maneuver(ManeuverType.MAINTAIN, 0, throttle=1.0, evade="dodge")
        assert sim._order_burn(ship, m, 1.0, 1.0) is None


class TestTheExecutionStateRecordsTheSwing:

    def test_swing_flag_on_execution_state(self):
        sim, me = _under_fire()
        _fly(sim, me, 90)
        states = [e.data for e in sim.events
                  if e.event_type == SimulationEventType.EXECUTION_STATE and e.ship_id == "me"]
        swung = [d for d in states if d.get("swing")]
        assert swung and all(d["mode"] in EVASION_OVERLAY_MODES for d in swung)
        assert all(d["applied_throttle"] == pytest.approx(d["requested_throttle"]) for d in swung)
