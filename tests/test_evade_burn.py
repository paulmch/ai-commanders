"""
The evade overlay on a burning order (owner's rule, pass 5; the rule is
written down in the src/simulation.py module docstring): "with evade the
engine burns the whole time using thrust vectoring to turn".

- the drive never runs below what the order burns while evasion has the helm,
  except with the nose more than EVADE_BURN_AGAINST_ORDER_DEG off the burn
  line (it would push against the order: the order's token turn throttle);
- the thrust axis stays inside EVADE_BURN_CONE_DEG of the burn line, so most
  of the push stays along it (cos 20 deg = 94 % at the cone edge); off the
  cone the hull turns toward it, it never holds an attitude off it;
- the swing runs under fire whatever the fire's bearing (along the burn line
  too), and a dodging charger is hit less than one that does not dodge;
- a HOLD_RANGE with the overlay keeps its range nearly as well as without;
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
    EVADE_BURN_AGAINST_ORDER_DEG,
    EVADE_BURN_CONE_DEG,
    EVADE_BURN_CYCLE_MAX_S,
    EVADE_BURN_CYCLE_MIN_S,
    EVADE_BURN_CYCLE_SHARE,
    EVASION_OVERLAY_MODES,
    HEADING_BURN_CONE_DEG,
    MANEUVER_TURN_THROTTLE,
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
        # Once the guns are gone the order flies; a nose the swing left off
        # the line swings back at the order's throttle (no turn-first token).
        sim, me = _under_fire(hull="corvette")
        for _ in range(120):
            sim.step()
            if sim.current_time > 20 and me.forward.angle_to(LINE) > math.radians(15.0):
                break
        for sid in ("s0", "s1", "s2"):
            sim.ships[sid].kinematic_state.position = Vector3D(5e6, 0.0, 0.0)
        sim.projectiles.clear()
        rows = _fly(sim, me, 20)
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


def _orders():
    """Burning orders the overlay can ride on (name, maneuver, setup)."""
    def heading(**kw):
        return Maneuver(ManeuverType.HEADING, 0, duration=1e9, throttle=1.0,
                        heading_direction={"x": 0.0, "y": 1.0, "z": 0.0}, evade="dodge", **kw)
    return [
        ("intercept", Maneuver(ManeuverType.INTERCEPT, 0, duration=1e9, throttle=1.0,
                               target_id="s0", evade="dodge"), None),
        ("standoff", Maneuver(ManeuverType.INTERCEPT, 0, duration=1e9, throttle=1.0,
                              target_id="s0", standoff_m=40_000.0, evade="dodge"), None),
        ("hold_range_open", Maneuver(ManeuverType.HOLD_RANGE, 0, duration=1e9, throttle=1.0,
                                     target_id="s0", hold_range_m=320_000.0, evade="dodge"), None),
        ("formation", heading(burn_start_time=25.0), None),
        ("brake", Maneuver(ManeuverType.BRAKE, 0, duration=1e9, throttle=1.0, evade="dodge"),
         lambda me: setattr(me.kinematic_state, "velocity", Vector3D(0.0, -4000.0, 0.0))),
        ("heading_90_off", heading(),
         lambda me: setattr(me.kinematic_state, "forward", Vector3D(1.0, 0.0, 0.0))),
    ]


class TestEveryBurningOrder:

    @pytest.mark.parametrize("name, maneuver, setup", _orders(), ids=[o[0] for o in _orders()])
    def test_evasion_never_flies_below_what_the_order_burns(self, monkeypatch, name, maneuver,
                                                            setup):
        sim, me = _under_fire(maneuver=maneuver)
        if setup:
            setup(me)
        flown = []
        original = CombatSimulation._swing_burn

        def wrapped(self, ship, maneuver_, burn, *args, **kwargs):
            off = ship.forward.angle_to(burn[0])
            out = original(self, ship, maneuver_, burn, *args, **kwargs)
            flown.append((self.current_time, burn[1], out, off))
            return out

        monkeypatch.setattr(CombatSimulation, "_swing_burn", wrapped)
        _fly(sim, me, 100)
        assert len(flown) >= 5, name
        for t, ordered, out, off in flown:
            if off > math.radians(EVADE_BURN_AGAINST_ORDER_DEG):
                # it would push against the order: the order's turn token
                assert out == pytest.approx(ordered * MANEUVER_TURN_THROTTLE), (name, t)
            else:
                assert out >= ordered - 1e-12, (name, t, ordered, out)
        if name == "heading_90_off":
            # the first turn under fire burns, and is recorded as off the line
            states = [e.data for e in sim.events
                      if e.event_type == SimulationEventType.EXECUTION_STATE and e.ship_id == "me"]
            assert any(d.get("swing") and d.get("off_line") for d in states)
        if name == "formation":
            # held until its start: an engine-off order for the overlay
            assert min(t for t, *_ in flown) >= 25.0


def _hold_range(hull, evade, seed):
    """The review's HOLD_RANGE case: hold 150 km on a torpedo cruiser
    launching every 10 s, a battleship firing every gun; RMS range error
    (km) over 240 s."""
    import random
    rng = random.Random(f"{hull}-{seed}")
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=seed)
    me = create_ship_from_fleet_data(
        "me", hull, "alpha", FLEET, position=Vector3D(0, 0, 0),
        velocity=Vector3D(rng.uniform(-300, 300), rng.uniform(-300, 300), 0),
        forward=Vector3D(1, 0, 0))
    sim.add_ship(me)
    tb = create_ship_from_fleet_data("tb", "cruiser_torpedo", "beta", FLEET,
                                     position=Vector3D(150e3, rng.uniform(-30e3, 30e3), 0),
                                     forward=Vector3D(-1, 0, 0))
    sim.add_ship(tb)
    tb.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, throttle=0.0,
                                   target_id="me")
    gun = create_ship_from_fleet_data("gun", "battleship", "beta", FLEET,
                                      position=Vector3D(140e3, rng.uniform(-60e3, 60e3), 0),
                                      forward=Vector3D(-1, 0, 0))
    sim.add_ship(gun)
    gun.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, throttle=0.0,
                                    target_id="me")
    gun.weapons_orders = {
        slot: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, slot, "me", min_hit_probability=0.0)
        for slot in gun.weapons if not slot.startswith("pd_")}
    me.current_maneuver = Maneuver(ManeuverType.HOLD_RANGE, 0, duration=1e9, throttle=1.0,
                                   target_id="tb", hold_range_m=150e3, evade=evade)
    err2 = n = 0
    while sim.current_time < 240 and not (me.is_destroyed or me.is_dying):
        if int(sim.current_time) % 10 == 0:
            sim.inject_command("tb", {"type": "launch_torpedo", "target_id": "me"})
        sim.step()
        e = ((tb.position - me.position).magnitude - 150e3) / 1000.0
        err2 += e * e
        n += 1
    return math.sqrt(err2 / max(n, 1))


class TestNoAttitudeHeldOffTheCone:

    def test_off_the_cone_the_hull_turns_toward_it(self):
        # A HEADING 180 deg off its order under fire: every off-line swing
        # step closes on the cone (never holds), and while the nose is more
        # than EVADE_BURN_AGAINST_ORDER_DEG off the drive does not push
        # against the order.
        sim, me = _under_fire(maneuver=Maneuver(
            ManeuverType.HEADING, 0, duration=1e9, throttle=1.0,
            heading_direction={"x": 0.0, "y": 1.0, "z": 0.0}, evade="dodge"))
        me.kinematic_state.forward = Vector3D(0.0, -1.0, 0.0)
        prev = None
        steps = 0
        for _ in range(60):
            sim.step()
            off = me.forward.angle_to(LINE)
            if me.evade_off_line and off > CONE + math.radians(25.0):
                steps += 1
                if prev is not None:
                    assert off <= prev + math.radians(0.5), (sim.current_time, math.degrees(off))
                if off > math.radians(EVADE_BURN_AGAINST_ORDER_DEG) + 0.05:
                    assert me.applied_throttle <= MANEUVER_TURN_THROTTLE + 1e-9
            prev = off if (me.evade_off_line and off > CONE + math.radians(25.0)) else None
        assert steps >= 5

    def test_hold_range_with_the_overlay_keeps_its_range(self):
        # The review measured 34-43 km RMS range error with an armour pick
        # holding the nose off the cone (11-22 km on the older overlay, 2-5 km
        # with no overlay). Now: within 3 km of no overlay.
        for seed in (1, 2, 3):
            plain = _hold_range("destroyer", None, seed)
            evading = _hold_range("destroyer", "full", seed)
            assert evading <= plain + 3.0, (seed, plain, evading)


def _charge(hull, evade, seed):
    """The review's nose-on charge: INTERCEPT at a battleship 180-260 km off
    with a destroyer beside it, both firing every gun. Returns (shots, hits,
    swung steps, swung steps with the nose > 5 deg off the line)."""
    import random
    rng = random.Random(f"{hull}-{seed}")
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=seed)
    R = rng.uniform(180e3, 260e3)
    me = create_ship_from_fleet_data("me", hull, "alpha", FLEET, position=Vector3D(0, 0, 0),
                                     velocity=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    sim.add_ship(me)
    for i, st in enumerate(("battleship", "destroyer")):
        a = math.radians((i - 0.5) * rng.uniform(4, 15))
        p = Vector3D(math.cos(a), math.sin(a), 0) * R
        s = create_ship_from_fleet_data(f"s{i}", st, "beta", FLEET, position=p,
                                        forward=(me.position - p).normalized())
        sim.add_ship(s)
        s.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, throttle=0.0,
                                      target_id="me")
        s.weapons_orders = {k: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, k, "me",
                                            min_hit_probability=0.0)
                            for k in s.weapons if not k.startswith("pd_")}
    me.current_maneuver = Maneuver(ManeuverType.INTERCEPT, 0, duration=1e9, throttle=1.0,
                                   target_id="s0", evade=evade)
    shots = hits = swung = off5 = 0
    while sim.current_time < 150 and not (me.is_destroyed or me.is_dying):
        for ev in sim.step():
            if ev.target_id == "me" and ev.event_type == SimulationEventType.PROJECTILE_LAUNCHED:
                shots += 1
            elif ev.target_id == "me" and ev.event_type == SimulationEventType.PROJECTILE_IMPACT:
                hits += 1
        if me.evade_swing and me.evade_burn is not None:
            swung += 1
            off5 += me.forward.angle_to(me.evade_burn[2]) > math.radians(5.0)
    return shots, hits, swung, off5


class TestFireAlongTheLineIsDodged:

    def test_the_axis_swings_under_fire_from_ahead(self):
        _, _, swung, off5 = _charge("destroyer", "dodge", 3)
        assert swung >= 50 and off5 >= 0.7 * swung

    def test_a_dodging_charger_is_hit_less(self):
        totals = {}
        for evade in (None, "dodge"):
            shots = hits = 0
            for seed in range(1, 5):
                s, h, _, _ = _charge("destroyer", evade, seed)
                shots += s
                hits += h
            totals[evade] = hits / shots
        # measured over seeds 1-8: 49 % without the overlay, 25 % with it
        assert totals["dodge"] < 0.7 * totals[None], totals


class TestEachHullCanFlyItsSwing:

    def test_cone_and_cycle_come_from_the_hull(self):
        sim = CombatSimulation(time_step=1.0, seed=1)
        seen = {}
        for cls in FLEET["ships"]:
            ship = create_ship_from_fleet_data(cls, cls, "alpha", FLEET, position=Vector3D(0, 0, 0))
            cone, cycle = sim._evade_burn_swing(ship)
            assert 0.0 < cone <= CONE + 1e-12
            edge_to_edge = sim._estimate_turn_time(ship, 2.0 * cone, engines_on=True)
            # the hull can cross its cone within EVADE_BURN_CYCLE_MAX_S, and it
            # is retargeted every SHARE of that (never settled on one aim)
            assert edge_to_edge <= EVADE_BURN_CYCLE_MAX_S + 1e-6
            assert cycle == pytest.approx(max(EVADE_BURN_CYCLE_MIN_S,
                                              EVADE_BURN_CYCLE_SHARE * edge_to_edge), abs=0.05)
            seen[cls] = (math.degrees(cone), cycle)
        # a corvette swings the full cone on the shortest cycle; a dreadnought
        # cannot cross 40 degrees in 30 s and swings a narrower cone
        assert seen["corvette"] == pytest.approx((EVADE_BURN_CONE_DEG, EVADE_BURN_CYCLE_MIN_S), abs=0.1)
        assert seen["dreadnought"][0] < EVADE_BURN_CONE_DEG
        assert seen["dreadnought"][1] == pytest.approx(EVADE_BURN_CYCLE_SHARE * EVADE_BURN_CYCLE_MAX_S)
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
