"""
Weapon arcs and turret traverse (src/weapon_arcs.py, CombatSimulation._arcs_*).

Scenario geometry comes from a small mount table built here (a destroyer with a
dorsal-hemisphere battery, a port-half and a starboard-half PD turret), never
from the real data/hull_mounts.json, whose placement may change: the real file
is only checked for completeness and loaded as-is.
"""

import json
import math

import pytest

from src import weapon_arcs as wa
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    TorpedoInFlight,
    WeaponsCommand,
    WeaponsOrder,
    create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo, TorpedoSpecs

with open("data/fleet_ships.json") as _f:
    FLEET = json.load(_f)

TRIDENT = TorpedoSpecs.from_fleet_data(
    warhead_yield_gj=0.0, penetrator_mass_kg=250.0, ammo_mass_kg=3600.0,
    exhaust_velocity_kps=8.0)


def _mask(predicate) -> str:
    """Mask from predicate(az_deg, el_deg) -> blocked, at the cell centres."""
    cells = []
    for ie in range(wa.EL_CELLS):
        for ia in range(wa.AZ_CELLS):
            cells.append(1 if predicate(-180 + 5 * ia + 2.5, -90 + 5 * ie + 2.5) else 0)
    return wa.encode_mask(cells)


def _table(dorsal_gun=lambda az, el: el < 0,
           pd0=lambda az, el: az < 0, pd1=lambda az, el: az > 0):
    """Fixture mount table for the destroyer (slots as in fleet_ships.json)."""
    return wa.parse_hull_mounts({
        "schema": 1,
        "grid": {"az_step_deg": 5, "el_step_deg": 5, "az_cells": 72, "el_cells": 36},
        "hulls": {"destroyer": {"length_m": 125, "mounts": [
            {"slot": "weapon_0", "kind": "spinal", "pos": [60, 0, 0], "axis": [1, 0, 0],
             "blocked": _mask(lambda az, el: math.hypot(az, el) > 30), "open_fraction": 0.07},
            {"slot": "weapon_1", "kind": "gun", "pos": [12, 0, 9], "axis": [0, 0, 1],
             "blocked": _mask(dorsal_gun), "open_fraction": 0.5},
            {"slot": "pd_0", "kind": "pd", "pos": [-20, 9, 0], "axis": [0, 1, 0],
             "blocked": _mask(pd0), "open_fraction": 0.5},
            {"slot": "pd_1", "kind": "pd", "pos": [-20, -9, 0], "axis": [0, -1, 0],
             "blocked": _mask(pd1), "open_fraction": 0.5},
        ]}}})


ALL_OPEN = dict(dorsal_gun=lambda az, el: False, pd0=lambda az, el: False,
                pd1=lambda az, el: False)


def _sim(table=None, arcs=True, roll=False, seed=5):
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed,
                           weapon_arcs_enabled=arcs, roll_to_unmask=roll)
    sim.echo = False
    if table is not None:
        sim.hull_mounts = table
    return sim


def _ship(sim, ship_id, faction, position, forward=Vector3D(1, 0, 0), hull="destroyer"):
    ship = create_ship_from_fleet_data(ship_id, hull, faction, FLEET,
                                       position=position, forward=forward)
    sim.add_ship(ship)
    return ship


def _fire_at(ship, target_id, slots=("weapon_1",)):
    ship.weapons_orders = {
        slot: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, slot, target_id, min_hit_probability=0.0)
        for slot in slots}


def _launches(sim, ship_id):
    return [e for e in sim.events
            if e.event_type == SimulationEventType.PROJECTILE_LAUNCHED and e.ship_id == ship_id]


def _torpedo(sim, torpedo_id, position, velocity, target_id, source_id):
    torp = Torpedo(specs=TRIDENT, position=position, velocity=velocity, target_id=target_id)
    torp.armed = True
    torp.fuel_exhausted = True
    torp.guidance_mode = GuidanceMode.COAST
    flight = TorpedoInFlight(torpedo_id=torpedo_id, torpedo=torp, source_ship_id=source_id)
    sim.torpedoes.append(flight)
    return flight


# =============================================================================
# Data and lookup
# =============================================================================

class TestMountTable:

    def test_every_weapon_slot_of_every_hull_has_a_mount_in_fleet_order(self):
        """An unlisted slot would silently fire all round (the ungated
        fallback for hull classes without mounts must not hide a real hull)."""
        table = wa.load_hull_mounts()
        for hull, spec in FLEET["ships"].items():
            assert hull in table, hull
            slots = [w["slot"] for w in spec["weapons"]]
            assert [m.slot for m in table[hull].mounts] == slots, hull
            for mount, weapon in zip(table[hull].mounts, spec["weapons"]):
                wtype = FLEET["weapon_types"].get(weapon["type"], {})
                if weapon["type"] == "pd_laser":
                    assert mount.kind == "pd"
                elif weapon["type"] == "torpedo_launcher":
                    assert mount.kind == "launcher" and not any(mount.mask)
                else:
                    assert mount.kind == ("gun" if wtype.get("is_turreted", True) else "spinal")
                assert len(mount.mask) == wa.AZ_CELLS * wa.EL_CELLS

    def test_cell_lookup_follows_the_contract(self):
        def d(az, el):
            a, e = math.radians(az), math.radians(el)
            return math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)
        assert wa.cell_index(*d(-180 + 2.5, -90 + 2.5)) == 0
        assert wa.cell_index(*d(2.5, 2.5)) == 18 * 72 + 36          # just off the nose, up-port
        assert wa.cell_index(*d(92.5, 2.5)) == 18 * 72 + 54         # port beam
        assert wa.cell_index(*d(-92.5, -2.5)) == 17 * 72 + 17       # starboard beam, below
        assert wa.cell_index(-1.0, 0.0, 0.0) == 18 * 72 + 0         # az +180 wraps to cell 0
        assert wa.cell_index(0.0, 0.0, 1.0) // 72 == 35             # zenith row
        cells = [(i * 7) % 3 == 0 for i in range(72 * 36)]
        assert list(wa.decode_mask(wa.encode_mask(cells))) == [int(c) for c in cells]

    def test_traverse_rates_are_named_per_weapon_class(self):
        assert wa.PD_TRAVERSE_DEG_S > wa.LIGHT_BATTERY_TRAVERSE_DEG_S \
            > wa.BATTERY_TRAVERSE_DEG_S > wa.HEAVY_BATTERY_TRAVERSE_DEG_S > 0
        # a heavy battery takes "a few seconds" through 90 degrees
        assert 3.0 <= 90.0 / wa.HEAVY_BATTERY_TRAVERSE_DEG_S <= 10.0
        # every trainable weapon type in the fleet data has a rate
        for name, spec in FLEET["weapon_types"].items():
            if spec.get("is_turreted", True) and name != "torpedo_launcher":
                assert name in wa.TRAVERSE_DEG_S_BY_WEAPON, name
        assert "pd_laser" in wa.TRAVERSE_DEG_S_BY_WEAPON


class TestTraverse:

    def _mount(self, rate=30.0, blocked=lambda az, el: False):
        table = _table(dorsal_gun=blocked)
        return wa.MountState(table["destroyer"].by_slot["weapon_1"], rate)

    def test_first_designation_is_free_then_the_mount_slews_at_its_rate(self):
        m = self._mount(rate=30.0)
        assert m.train(1.0, 0.0, 0.0, 1.0) == wa.ON_TARGET          # trained before battle
        # swing 90 degrees to port at 30 deg/s: slewing for two ticks, on in the third
        states = [m.train(0.0, 1.0, 0.0, 1.0) for _ in range(3)]
        assert states == [wa.SLEWING, wa.SLEWING, wa.ON_TARGET]
        # along the great circle: stays in the x-y plane on the way
        m2 = self._mount(rate=30.0)
        m2.train(1.0, 0.0, 0.0, 1.0)
        m2.train(0.0, 1.0, 0.0, 1.0)
        az, el = wa.az_el_deg(m2.ax, m2.ay, m2.az)
        assert az == pytest.approx(30.0, abs=1e-6) and el == pytest.approx(0.0, abs=1e-6)

    def test_masked_target_trains_to_the_edge_and_waits(self):
        # open only on the port side (az > 0)
        m = self._mount(rate=30.0, blocked=lambda az, el: az < 0)
        assert m.train(0.0, 1.0, 0.0, 1.0) == wa.ON_TARGET
        # target swings to the starboard quarter: masked
        want = (math.cos(math.radians(-40)), math.sin(math.radians(-40)), 0.0)
        for _ in range(6):
            assert m.train(*want, 1.0) == wa.MASKED
        az, _ = wa.az_el_deg(m.ax, m.ay, m.az)
        assert 0.0 <= az <= 5.0            # resting on the last open cell
        assert m.bears(m.ax, m.ay, m.az)   # never points into its own hull
        # target comes back into the arc: one tick to get back on it
        assert m.train(math.cos(0.3), math.sin(0.3), 0.0, 1.0) == wa.ON_TARGET


# =============================================================================
# Gun gating in the engine
# =============================================================================

class TestGunGating:

    def _duel(self, target_z_km, roll=False, table=None, arcs=True):
        sim = _sim(_table() if table is None else table, arcs=arcs, roll=roll)
        shooter = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "beta", Vector3D(100_000, 0, target_z_km * 1000.0),
              forward=Vector3D(-1, 0, 0))
        _fire_at(shooter, "b")
        return sim, shooter

    def test_a_battery_fires_at_a_target_its_hull_leaves_open(self):
        sim, shooter = self._duel(target_z_km=40)
        for _ in range(5):
            sim.step()
        assert _launches(sim, "a")
        assert shooter.arcs.guns["weapon_1"].state == wa.ON_TARGET

    def test_a_masked_battery_holds_fire_keeps_its_ammo_and_reports_it(self):
        sim, shooter = self._duel(target_z_km=-40)
        ammo = shooter.weapons["weapon_1"].ammo_remaining
        for _ in range(12):
            sim.step()
        assert not _launches(sim, "a")
        assert shooter.weapons["weapon_1"].ammo_remaining == ammo
        mount = shooter.arcs.guns["weapon_1"]
        assert mount.state == wa.MASKED
        assert mount.bears(mount.ax, mount.ay, mount.az)        # resting at the edge
        status = [e.data["reason"] for e in sim.events
                  if e.event_type == SimulationEventType.WEAPON_STATUS
                  and e.data.get("weapon_slot") == "weapon_1"]
        assert "masked" in status
        masked = [e for e in sim.events if e.event_type == SimulationEventType.MOUNT_MASKED]
        assert len(masked) == 1                                  # throttled: once on entry
        assert masked[0].data["slot"] == "weapon_1" and masked[0].data["masked"] is True
        assert masked[0].target_id == "b" and masked[0].data["reason"] == "hull"
        assert sim.arcs_tally["masked"] > 0 and sim.arcs_tally["released"] >= sim.arcs_tally["masked"]

    def test_arcs_off_fires_all_round(self):
        sim, _ = self._duel(target_z_km=-40, arcs=False)
        for _ in range(5):
            sim.step()
        assert _launches(sim, "a")

    def test_a_hull_class_without_mounts_is_ungated(self):
        sim, shooter = self._duel(target_z_km=-40, table={})
        for _ in range(5):
            sim.step()
        assert _launches(sim, "a") and not shooter.arcs.mounts

    def test_slewing_delays_the_round_until_the_mount_arrives(self):
        """Two enemies 90 degrees apart (both in the open hemisphere): after a
        retarget the 30 deg/s battery needs 3 s before the next round leaves."""
        sim = _sim(_table(dorsal_gun=lambda az, el: False))
        shooter = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "beta", Vector3D(100_000, 0, 0), forward=Vector3D(-1, 0, 0))
        _ship(sim, "c", "beta", Vector3D(0, 100_000, 0), forward=Vector3D(0, -1, 0))
        shooter.primary_target_id = "b"        # trained on b, holding fire
        sim.step()
        assert shooter.arcs.guns["weapon_1"].state == wa.ON_TARGET and not _launches(sim, "a")
        _fire_at(shooter, "c")
        states = []
        for _ in range(4):
            before = len(_launches(sim, "a"))
            sim.step()
            states.append((shooter.arcs.guns["weapon_1"].state, len(_launches(sim, "a")) - before))
        expected_slew_ticks = math.ceil(90.0 / wa.BATTERY_TRAVERSE_DEG_S) - 1
        assert [s for s, _ in states[:expected_slew_ticks]] == [wa.SLEWING] * expected_slew_ticks
        assert all(n == 0 for _, n in states[:expected_slew_ticks])
        assert states[expected_slew_ticks] == (wa.ON_TARGET, 1)

    def test_the_spinal_keeps_its_analytic_gimbal_rule(self):
        """pivot_range_deg decides; the mount only mirrors it (state 2 / 3)."""
        pivot = FLEET["weapon_types"]["spinal_coiler_mk3"]["pivot_range_deg"]
        for off_deg, fires in ((pivot - 8, True), (2 * pivot + 10, False)):
            rows = {}
            for arcs in (True, False):
                sim = _sim(_table(), arcs=arcs)
                shooter = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
                a = math.radians(off_deg)
                _ship(sim, "b", "beta",
                      Vector3D(100_000 * math.cos(a), 100_000 * math.sin(a), 0),
                      forward=Vector3D(-1, 0, 0))
                _fire_at(shooter, "b", slots=("weapon_0",))
                for _ in range(3):
                    sim.step()
                rows[arcs] = len(_launches(sim, "a"))
                if arcs:
                    state = shooter.arcs.guns["weapon_0"].state
                    assert state == (wa.ON_TARGET if fires else wa.MASKED)
            assert rows[True] == rows[False] and (rows[True] > 0) == fires


# =============================================================================
# Roll to unmask
# =============================================================================

class TestRollToUnmask:

    def _masked_duel(self, roll):
        sim = _sim(_table(), roll=roll)
        shooter = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        shooter.current_maneuver = Maneuver(ManeuverType.HEADING, 0, duration=1e9, throttle=0.6,
                                            heading_direction={"x": 0.6, "y": 0.8, "z": 0.0})
        target = _ship(sim, "b", "beta", Vector3D(150_000, 0, 0), forward=Vector3D(-1, 0, 0))
        return sim, shooter, target

    def test_roll_brings_a_masked_battery_to_bear_rate_limited(self):
        sim, shooter, _ = self._masked_duel(roll=True)
        sim.step()                      # t=0: no gun target yet, nothing to settle on
        # the target drops below the deck: masked from here on without a roll
        sim.ships["b"].kinematic_state.position = Vector3D(120_000, 0, -90_000)
        _fire_at(shooter, "b")
        accel, max_rate = sim._arcs_roll_params(shooter)
        rolls = [shooter.arcs.roll]
        first_shot_at = None
        for _ in range(200):
            sim.step()
            rolls.append(shooter.arcs.roll)
            if first_shot_at is None and _launches(sim, "a"):
                first_shot_at = sim.current_time
        steps = [abs(b - a) for a, b in zip(rolls, rolls[1:])]
        assert max(steps) <= max_rate * 1.0 + 1e-9                 # never faster than the cap
        # ...and the rate itself changes by the RCS roll acceleration (the
        # last tick of braking onto the goal may take two ticks' worth)
        assert max(abs(b - a) for a, b in zip(steps, steps[1:])) <= 2.0 * accel * 1.0 + 1e-9
        assert abs(shooter.arcs.roll) > math.radians(10)            # it did roll
        assert first_shot_at is not None and first_shot_at > 5.0    # ...and it took time
        cleared = [e for e in sim.events if e.event_type == SimulationEventType.MOUNT_MASKED
                   and e.data["masked"] is False]
        assert cleared and cleared[0].data["duration_s"] > wa.MOUNT_MASKED_REPORT_S
        # hysteresis: once the battery bears the roll goal stands still
        goal = shooter.arcs.roll_goal
        for _ in range(40):
            sim.step()
        assert shooter.arcs.roll_goal == goal

    def test_roll_never_changes_motion_or_the_navigation_frame(self):
        """Same scenario with and without the roll: positions, velocities,
        forward and the navigation up evasion steers by are bit-identical."""
        tracks = {}
        for roll in (True, False):
            sim, shooter, target = self._masked_duel(roll=roll)
            target.current_maneuver = Maneuver(ManeuverType.EVASIVE, 0, duration=1e9, throttle=1.0)
            sim.step()
            sim.ships["b"].kinematic_state.position = Vector3D(120_000, 0, -90_000)
            shooter.primary_target_id = "b"     # mounts track, nothing fires
            rows = []
            for _ in range(120):
                sim.step()
                for s in (shooter, target):
                    ks = s.kinematic_state
                    rows.append((ks.position.to_tuple(), ks.velocity.to_tuple(),
                                 ks.forward.to_tuple(), ks.up.to_tuple()))
            tracks[roll] = rows
            if roll:
                assert abs(shooter.arcs.roll) > math.radians(10)
        assert tracks[True] == tracks[False]

    def test_inbound_torpedoes_outrank_the_gun_target(self):
        """The gun wants the target kept above the deck; a torpedo comes in on
        a bearing only a roll the other way opens for a PD turret: the PD wins."""
        # PD turrets see only a narrow dorsal fan; the gun sees the dorsal hemisphere
        table = _table(pd0=lambda az, el: el < 50, pd1=lambda az, el: el < 50)
        sim = _sim(table, roll=True)
        shooter = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "beta", Vector3D(0, 150_000, 60_000), forward=Vector3D(0, -1, 0))
        _fire_at(shooter, "b")
        sim.step()
        assert shooter.arcs.guns["weapon_1"].state == wa.ON_TARGET
        # torpedo from below-starboard: needs ~180 degrees of roll for the PD fan
        _torpedo(sim, "t", Vector3D(0, -60_000, -200_000), Vector3D(0, 0, 0), "a", "b")
        for _ in range(6):
            sim.step()
        phi = shooter.arcs.roll_goal - shooter.arcs.roll
        torps, pd_mounts, guns = sim._arcs_roll_inputs(shooter, shooter.arcs)
        assert torps and pd_mounts
        assert sim._arcs_roll_score(phi, torps, pd_mounts, guns)[0] \
            > sim._arcs_roll_score(0.0, torps, pd_mounts, guns)[0]
        assert abs(shooter.arcs.roll_goal) > math.radians(60)


# =============================================================================
# Point defence
# =============================================================================

class TestPointDefenceArcs:

    def _approach(self, side_y_m, table=None, arcs=True, ticks=40):
        sim = _sim(_table() if table is None else table, arcs=arcs)
        _ship(sim, "d", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "e", "beta", Vector3D(0, 900_000, 0), forward=Vector3D(0, -1, 0))
        sign = 1.0 if side_y_m > 0 else -1.0
        flight = _torpedo(sim, "t", Vector3D(0, side_y_m, 0),
                          Vector3D(0, -sign * 4000.0, 0), "d", "e")
        for _ in range(ticks):
            sim.step()
            if flight not in sim.torpedoes:
                break
        engaged = {e.data.get("turret_name") or e.data.get("turret")
                   for e in sim.events if e.event_type == SimulationEventType.PD_ENGAGED}
        return sim, flight, engaged

    def test_only_the_turret_that_bears_engages(self):
        sim, flight, engaged = self._approach(side_y_m=200_000, ticks=12)       # from port
        states = [m.state for m in sim.ships["d"].arcs.pd]
        assert flight.heat_absorbed_j > 0
        assert states[0] == wa.ON_TARGET and states[1] == wa.MASKED
        sim2, flight2, _ = self._approach(side_y_m=-200_000, ticks=12)          # from starboard
        states2 = [m.state for m in sim2.ships["d"].arcs.pd]
        assert states2[1] == wa.ON_TARGET and states2[0] == wa.MASKED
        # one turret of two: half the energy of the unmasked pair
        _, both, _ = self._approach(side_y_m=200_000, arcs=False, ticks=12)
        assert not both.is_disabled
        assert flight.heat_absorbed_j == pytest.approx(0.5 * both.heat_absorbed_j, rel=0.02)
        assert flight2.heat_absorbed_j == pytest.approx(flight.heat_absorbed_j, rel=1e-9)

    def test_a_turret_that_bears_delivers_exactly_what_it_did_without_arcs(self):
        """The doctrine's per-turret figures (one turret blinds ~5 km/s of
        closure, turrets needed per round) are measured with every turret
        dwelling; PD_TRAVERSE_DEG_S is fast enough that a turret that BEARS
        loses no dwell to traverse - same energy on the torpedo, tick for tick."""
        _, with_arcs, _ = self._approach(side_y_m=260_000, table=_table(**ALL_OPEN), ticks=70)
        _, without, _ = self._approach(side_y_m=260_000, arcs=False, ticks=70)
        assert with_arcs.heat_absorbed_j > 0
        assert with_arcs.heat_absorbed_j == without.heat_absorbed_j
        assert with_arcs.is_disabled == without.is_disabled

    def test_a_turret_switches_between_the_rounds_of_a_salvo_without_losing_dwell(self):
        """Two torpedoes 20 degrees apart: inside one tick of traverse."""
        assert wa.PD_TRAVERSE_DEG_S * 1.0 >= 90.0
        m = wa.MountState(_table(**ALL_OPEN)["destroyer"].by_slot["pd_0"], wa.PD_TRAVERSE_DEG_S)
        m.train(0.0, 1.0, 0.0, 1.0)
        a = math.radians(20.0)
        assert m.train(math.sin(a), math.cos(a), 0.0, 1.0) == wa.ON_TARGET
        assert m.train(0.0, -1.0, 0.0, 1.0) == wa.SLEWING       # the far side costs one tick
        assert m.train(0.0, -1.0, 0.0, 1.0) == wa.ON_TARGET

    def test_savability_counts_only_turrets_that_bear(self):
        """A round that needs both turrets is savable without arcs and not
        savable when only one of the two bears on it."""
        for closure_kps in range(3, 30):
            sim = _sim(_table())
            d = _ship(sim, "d", "alpha", Vector3D(0, 0, 0))
            _ship(sim, "e", "beta", Vector3D(0, 900_000, 0), forward=Vector3D(0, -1, 0))
            _torpedo(sim, "t", Vector3D(0, 200_000, 0),
                     Vector3D(0, -closure_kps * 1000.0, 0), "d", "e")
            sim._arcs_begin_tick()
            with_arcs = sim._build_pd_target_list(d)[0]
            sim.weapon_arcs = False
            without = sim._build_pd_target_list(d)[0]
            assert with_arcs["turrets_needed_kill"] == without["turrets_needed_kill"]
            if without["turrets_needed_kill"] == 2:
                assert without["savable"] is True and with_arcs["savable"] is False
                break
            assert with_arcs["savable"] == without["savable"]
        else:
            pytest.fail("no closure in 3-29 km/s needs exactly two turrets")


class TestCommandersCountWhatBears:

    def _scene(self, arcs=True):
        sim = _sim(_table(), arcs=arcs)
        d = _ship(sim, "d", "alpha", Vector3D(0, 0, 0))
        e = _ship(sim, "e", "beta", Vector3D(0, 300_000, 0), forward=Vector3D(0, -1, 0))
        return sim, d, e

    def test_pd_cover_against_a_direction_counts_turrets_that_bear(self):
        from src.llm.fire_solutions import pd_cover
        sim, d, e = self._scene()
        assert pd_cover(sim, d, 250.0)[0] == 2                        # turrets fitted
        assert pd_cover(sim, d, 250.0, threat_from=e.position)[0] == 1    # port turret only
        off, d2, e2 = self._scene(arcs=False)
        assert pd_cover(off, d2, 250.0, threat_from=e2.position)[0] == 2

    def test_gun_lines_say_what_bears_and_what_unmasks_the_rest(self):
        from src.llm import arcs_status
        sim, d, e = self._scene()
        below = _ship(sim, "f", "beta", Vector3D(0, 100_000, -100_000), forward=Vector3D(0, -1, 0))
        d.primary_target_id = "f"
        sim.step()
        report = arcs_status.gun_report(sim, d, below)
        battery = next(r for r in report["mounts"] if r["slot"] == "weapon_1")
        assert battery["bears"] is False
        deg, secs = battery["unmask"]
        accel, rate = sim._arcs_roll_params(d)
        assert 30.0 <= deg <= 60.0                 # 45 degrees below the deck, 15 degree steps
        assert secs == pytest.approx(wa.roll_time_s(math.radians(deg), accel, rate))
        text = arcs_status.gun_arcs_text(sim, d, below)
        assert "0 of 2 gun mounts bear" in text and "weapon_1 masked by own hull" in text
        assert f"{deg:.0f} deg roll" in text
        assert arcs_status.mount_bears(sim, d, "weapon_1", below) is False
        above = _ship(sim, "g", "beta", Vector3D(100_000, 0, 40_000), forward=Vector3D(-1, 0, 0))
        assert arcs_status.mount_bears(sim, d, "weapon_1", above) is True
        assert "2 of 2 gun mounts bear" in arcs_status.gun_arcs_text(sim, d, above)
        lines = arcs_status.arcs_status_lines(sim, d)
        assert any(line.startswith("ARCS vs") for line in lines)
        # arcs off: nothing to report, every mount counts
        off, d2, e2 = self._scene(arcs=False)
        assert arcs_status.arcs_status_lines(off, d2) == []
        assert arcs_status.mount_bears(off, d2, "weapon_1", e2) is True


# =============================================================================
# Recording
# =============================================================================

class TestTraceFields:

    def test_up_and_tur_follow_the_contract(self):
        sim = _sim(_table(), roll=True)
        shooter = _ship(sim, "a", "alpha", Vector3D(0, 0, 0), forward=Vector3D(0.6, 0.8, 0.0))
        _ship(sim, "b", "beta", Vector3D(100_000, 0, -80_000), forward=Vector3D(-1, 0, 0))
        _fire_at(shooter, "b")
        for _ in range(30):
            sim.step()
        fields = sim.arcs_trace_fields(shooter)
        up, fwd = fields["up"], shooter.forward
        assert math.sqrt(sum(c * c for c in up)) == pytest.approx(1.0, abs=1e-3)
        assert up[0] * fwd.x + up[1] * fwd.y + up[2] * fwd.z == pytest.approx(0.0, abs=1e-3)
        assert len(fields["tur"]) == len(sim.hull_mounts["destroyer"].mounts)
        for az, el, state in fields["tur"]:
            assert -180.0 <= az <= 180.0 and -90.0 <= el <= 90.0 and state in (0, 1, 2, 3)
        # the recorded gun azimuth/elevation is the lead direction in the body frame
        mount = shooter.arcs.guns["weapon_1"]
        if mount.state == wa.ON_TARGET:
            world = shooter.arcs.to_world(mount.ax, mount.ay, mount.az)
            los = (sim.ships["b"].position - shooter.position).normalized()
            assert world[0] * los.x + world[1] * los.y + world[2] * los.z > 0.95
        # arcs off: no fields, old consumers see what they always saw
        off = _sim(_table(), arcs=False)
        s = _ship(off, "a", "alpha", Vector3D(0, 0, 0))
        off.step()
        assert off.arcs_trace_fields(s) == {}

    def test_arcs_never_touch_the_battle_rng(self):
        """Mounts, masks and roll are deterministic geometry: the battle RNG
        and the damage stream end in the same state with arcs on or off when
        nothing fires."""
        states = {}
        for arcs in (True, False):
            sim = _sim(_table(), arcs=arcs, roll=True, seed=11)
            a = _ship(sim, "a", "alpha", Vector3D(0, 0, 0))
            _ship(sim, "b", "beta", Vector3D(100_000, 0, -80_000), forward=Vector3D(-1, 0, 0))
            a.primary_target_id = "b"
            for _ in range(40):
                sim.step()
            states[arcs] = (sim.rng.getstate(), sim._damage_rng.getstate())
        assert states[True] == states[False]
