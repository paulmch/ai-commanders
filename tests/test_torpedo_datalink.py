"""
Torpedo launch profiles and the torpedo datalink (pass 5, lane retarget).

- profiles (src/torpedo.py): 'sprint' is the default and flies exactly as
  before profiles existed; 'reserve' cruises at RESERVE_CRUISE_CLOSING_KPS and
  keeps the rest of its tank until RESERVE_TERMINAL_WINDOW_S before impact.
- datalink (src/torpedo_link.py, CombatSimulation._torpedo_datalink): a
  side's live-seeker rounds reassign a lost (destroyed or dying) target as a
  group, spread by what each target needs (torpedo_link.rounds_needed), and
  divert reserve-profile surplus before the target dies; deterministic,
  sorted by ids, mirror-symmetric.

Bit-for-bit default: with the datalink switched off and no profile chosen
the engine reproduces the pre-profile pass-5 perf golden exactly (checked by
scripts/perf_check.py with AIC_TORPEDO_DATALINK=0 against a golden recorded
on the pass-5 base; see the lane report). Here: the sprint guidance and the
launch event are unchanged by the new fields.
"""

import contextlib
import io
import json
import math
import random
from pathlib import Path

import pytest

from src import torpedo as T
from src import torpedo_link as L
from src.physics import Vector3D
from src.simulation import (
    CombatSimulation,
    SimulationEventType,
    TorpedoInFlight,
    _build_torpedo_specs,
    create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "fleet_ships.json") as f:
    FLEET = json.load(f)
SPECS = _build_torpedo_specs(FLEET["weapon_types"]["torpedo_launcher"])


def _sim(seed=3):
    return CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)


def _ship(sim, sid, typ, fac, pos, vel=None, pd=False):
    s = create_ship_from_fleet_data(
        sid, typ, fac, FLEET, position=pos, velocity=vel or Vector3D(0, 0, 0),
        forward=Vector3D(-1, 0, 0) if fac == "beta" else Vector3D(1, 0, 0))
    if not pd:
        s.point_defense = []
    sim.add_ship(s)
    return s


def _round(sim, tid, target, dv, pos, vel, source="a", profile="sprint"):
    torp = Torpedo(specs=SPECS, position=pos, velocity=vel, target_id=target,
                   guidance_mode=GuidanceMode.COLLISION, profile=profile)
    torp.armed = True
    torp.current_mass_kg = SPECS.dry_mass_kg * math.exp(dv / SPECS.exhaust_velocity_kps)
    torp.remaining_delta_v_kps = dv
    tf = TorpedoInFlight(torpedo_id=tid, torpedo=torp, source_ship_id=source,
                         launch_time=sim.current_time)
    sim.torpedoes.append(tf)
    return tf


def _steps(sim, n):
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            sim.step()
            if not sim.torpedoes:
                break


def _rt(sim):
    return [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_RETARGETED]


def _launch(sim, shooter, target, profile=None):
    for l in shooter.torpedo_launchers:
        l.last_launch_time = -1e9
    cmd = {"target_id": target}
    if profile:
        cmd["profile"] = profile
    assert sim._handle_torpedo_command(shooter, cmd)
    return sim.torpedoes[-1]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles:

    def test_profile_constants(self):
        assert T.TORPEDO_PROFILES == ("sprint", "reserve")
        assert T.RESERVE_CRUISE_CLOSING_KPS == 8.0
        assert T.RESERVE_TERMINAL_WINDOW_S == 20.0
        assert T.normalize_torpedo_profile("RESERVE") == "reserve"
        assert T.normalize_torpedo_profile(None) == "sprint"
        assert T.normalize_torpedo_profile("warp") == "sprint"

    def test_default_launch_is_sprint_and_records_no_profile_field(self):
        sim = _sim()
        a = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "battlecruiser", "beta", Vector3D(500e3, 0, 0))
        tf = _launch(sim, a, "b")
        assert tf.torpedo.profile == "sprint"
        ev = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_LAUNCHED]
        assert "profile" not in ev[-1].data          # byte-identical to before
        tf2 = _launch(sim, a, "b", "reserve")
        assert tf2.torpedo.profile == "reserve"
        ev = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_LAUNCHED]
        assert ev[-1].data["profile"] == "reserve"

    def _fly(self, profile, range_km=500.0):
        sim = _sim()
        a = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0))
        _ship(sim, "b", "battlecruiser", "beta", Vector3D(range_km * 1e3, 0, 0))
        tf = _launch(sim, a, "b", profile)
        dv60 = None
        with contextlib.redirect_stdout(io.StringIO()):
            while sim.torpedoes and sim.current_time < 600:
                sim.step()
                if dv60 is None and sim.current_time >= 60:
                    dv60 = tf.torpedo.remaining_delta_v_kps
        hit = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_IMPACT]
        assert hit
        return hit[0].timestamp, hit[0].data["total_damage_gj"], dv60

    def test_reserve_keeps_its_tank_and_pays_in_time(self):
        """Measured (scripts/torpedo_profile_sweep.py): from 500 km sprint
        lands at ~72 s with ~1.4 km/s left at 60 s; reserve ~81 s with ~6 km/s
        left at 60 s, and its terminal burn brings the impact energy back to
        sprint's (+-5 %)."""
        t_s, gj_s, dv_s = self._fly("sprint")
        t_r, gj_r, dv_r = self._fly("reserve")
        assert dv_s < 2.0 and dv_r > 5.5
        assert t_r >= t_s + 5
        assert abs(gj_r - gj_s) / gj_s < 0.05

    def test_reserve_reach_after_reassignment_exceeds_sprint(self):
        def reach_hits(profile, offset_km):
            sim = _sim()
            a = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0))
            b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(800e3, 0, 0))
            tf = _launch(sim, a, "b1", profile)
            with contextlib.redirect_stdout(io.StringIO()):
                while tf.torpedo.position.distance_to(b1.position) > 200e3:
                    sim.step()
            b2 = _ship(sim, "b2", "battlecruiser", "beta",
                       Vector3D(b1.position.x, offset_km * 1e3, 0))
            b1.is_destroyed = True
            sim._reassign_torpedo(tf, b2, "target_lost", "probe")
            _steps(sim, 200)
            return any(e.event_type == SimulationEventType.TORPEDO_IMPACT for e in sim.events)
        # 200 km to go: sprint reaches ~22 km abeam, reserve ~110 km
        assert not reach_hits("sprint", 60)
        assert reach_hits("reserve", 60)


# ---------------------------------------------------------------------------
# Datalink: group reassignment
# ---------------------------------------------------------------------------

def _orphan_group(n=6, order=None, sx=1.0, faction="alpha", seed=3):
    """n live rounds whose battlecruiser target just died; two destroyers
    (PD off: each needs KILL_ROUNDS_LIGHT = 4) ahead, both reachable."""
    enemy = "beta" if faction == "alpha" else "alpha"
    sim = _sim(seed)
    ships = [("a", "corvette", faction, Vector3D(sx * -2_000e3, 0, 0)),
             ("dead", "battlecruiser", enemy, Vector3D(sx * 150e3, 0, 0)),
             ("d1", "destroyer", enemy, Vector3D(sx * 300e3, 15e3, 0)),
             ("d2", "destroyer", enemy, Vector3D(sx * 300e3, -25e3, 0))]
    for i in (order or range(len(ships))):
        _ship(sim, *ships[i])
    rounds = [( f"t{i:02d}", Vector3D(0, (i - n / 2) * 500.0, 0)) for i in range(n)]
    for tid, pos in rounds:
        _round(sim, tid, "dead", 4.0, pos, Vector3D(sx * 12_000, 0, 0))
    sim.get_ship("dead").is_destroyed = True
    return sim


# A destroyer with no PD (soak 0) against rounds closing at ~12 km/s
# (~18 GJ each, below the 22 GJ reference): kill_rounds_at(4, 18) = 5.
NEED_DD = L.kill_rounds_at(L.KILL_ROUNDS_LIGHT, 0.5 * 250.0 * 12_000.0 ** 2 / 1e9)


def _assignment(sim):
    return {e.data["torpedo_id"]: e.target_id for e in _rt(sim)}


class TestGroupReassignment:

    def test_lost_target_rounds_are_spread_by_need(self):
        sim = _orphan_group(6)
        _steps(sim, 1)
        plan = _assignment(sim)
        assert len(plan) == 6
        counts = {s: list(plan.values()).count(s) for s in ("d1", "d2")}
        assert counts["d1"] > 0 and counts["d2"] > 0
        assert max(counts.values()) == NEED_DD     # one filled to its need
        ev = _rt(sim)
        assert {e.data["reason"] for e in ev} == {"target_lost"}
        assert len({e.data["group"] for e in ev}) == 1
        assert all(e.data["group_size"] == 6 for e in ev)

    def test_pre_link_rule_sends_every_orphan_to_the_same_hull(self):
        sim = _orphan_group(6)
        sim.torpedo_datalink = False
        _steps(sim, 1)
        assert len(set(_assignment(sim).values())) == 1

    def test_rounds_already_inbound_count_against_the_need(self):
        sim = _orphan_group(4)
        # live rounds already homing on d1: d1 needs nothing more
        for i in range(NEED_DD):
            _round(sim, f"u{i}", "d1", 4.0, Vector3D(100e3, 14e3 + i * 100, 0),
                   Vector3D(12_000, 0, 0))
        _steps(sim, 1)
        plan = _assignment(sim)
        assert set(plan.values()) == {"d2"}

    def test_never_onto_friendlies_or_hulks(self):
        sim = _orphan_group(4)
        sim.get_ship("d1").is_dying = True
        _ship(sim, "a2", "destroyer", "alpha", Vector3D(300e3, 0, 0))
        _steps(sim, 1)
        assert set(_assignment(sim).values()) == {"d2"}

    def test_blinded_and_dry_rounds_stay_off_the_link(self):
        sim = _orphan_group(4)
        sim.torpedoes[0].is_disabled = True
        sim.torpedoes[1].torpedo.fuel_exhausted = True
        _steps(sim, 1)
        plan = _assignment(sim)
        assert "t00" not in plan and "t01" not in plan
        assert len(plan) == 2

    def test_reassignment_respects_the_time_scaled_reserve(self):
        """A round whose steer would leave less than
        reassign_reserve_kps(t) stays orphaned."""
        sim = _sim()
        _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
        dead = _ship(sim, "dead", "battlecruiser", "beta", Vector3D(150e3, 0, 0))
        d1 = _ship(sim, "d1", "destroyer", "beta", Vector3D(300e3, 25e3, 0))
        probe = Torpedo(specs=SPECS, position=Vector3D(0, 0, 0),
                        velocity=Vector3D(12_000, 0, 0), target_id="dead")
        probe.remaining_delta_v_kps = 1e6
        _, t, left = probe.can_intercept(d1.position, d1.velocity)
        steer = 1e6 - left
        reserve = L.reassign_reserve_kps(t, 1.12)
        _round(sim, "short", "dead", steer + 0.5 * reserve, Vector3D(0, 0, 0),
               Vector3D(12_000, 0, 0))
        _round(sim, "ok", "dead", steer + 1.5 * reserve, Vector3D(0, 0, 0),
               Vector3D(12_000, 0, 0))
        dead.is_destroyed = True
        _steps(sim, 1)
        assert _assignment(sim) == {"ok": "d1"}

    def test_reserve_curve(self):
        assert L.reassign_reserve_kps(0, 1.12) == pytest.approx(0.25)
        assert L.reassign_reserve_kps(20, 1.12) == pytest.approx(0.65)
        assert L.reassign_reserve_kps(100, 1.12) == pytest.approx(1.12)

    def test_order_independent(self):
        base = _assignment((lambda s: (_steps(s, 1), s)[1])(_orphan_group(6)))
        for order in ([3, 2, 1, 0], [2, 0, 3, 1]):
            sim = _orphan_group(6, order=order)
            random.Random(7).shuffle(sim.torpedoes)
            _steps(sim, 1)
            assert _assignment(sim) == base

    def test_mirror_symmetric(self):
        a = _orphan_group(6)
        _steps(a, 1)
        b = _orphan_group(6, sx=-1.0, faction="beta")
        _steps(b, 1)
        assert _assignment(a) == _assignment(b)

    def test_both_sides_in_one_battle_are_mirror_images(self):
        """Alpha's and beta's orphan groups in ONE sim (mirror images of each
        other), fleets added in either order: the same plan for both sides."""
        def run(order):
            sim = _sim()
            spec = []
            for fac, sx, en in (("alpha", 1.0, "beta"), ("beta", -1.0, "alpha")):
                p = "x" if fac == "alpha" else "y"
                spec.append([(f"{p}src", "corvette", fac, Vector3D(sx * -2_000e3, 0, 0)),
                             (f"{p}dead", "battlecruiser", en, Vector3D(sx * 150e3, 0, 0)),
                             (f"{p}d1", "destroyer", en, Vector3D(sx * 300e3, 15e3, 9e5 * sx)),
                             (f"{p}d2", "destroyer", en, Vector3D(sx * 300e3, -25e3, 9e5 * sx))])
            for i in order:
                for args in spec[i]:
                    _ship(sim, *args)
            for fac, sx, p in (("alpha", 1.0, "x"), ("beta", -1.0, "y")):
                for i in range(6):
                    _round(sim, f"{p}t{i}", f"{p}dead", 4.0,
                           Vector3D(0, (i - 3) * 500.0, 9e5 * sx), Vector3D(sx * 12_000, 0, 0),
                           source=f"{p}src")
            sim.get_ship("xdead").is_destroyed = True
            sim.get_ship("ydead").is_destroyed = True
            _steps(sim, 1)
            plan = _assignment(sim)
            return ({k[2:]: v[1:] for k, v in plan.items() if k.startswith("xt")},
                    {k[2:]: v[1:] for k, v in plan.items() if k.startswith("yt")})
        x1, y1 = run([0, 1])
        x2, y2 = run([1, 0])
        assert x1 == y1 == x2 == y2 and len(x1) == 6


# ---------------------------------------------------------------------------
# Datalink: surplus diversion (reserve rounds only)
# ---------------------------------------------------------------------------

def _surplus_sim(profile, dv=6.0):
    """Ten rounds inbound on one destroyer (needs 4), another destroyer
    reachable 40 km abeam; rounds ~25 s out."""
    sim = _sim()
    _ship(sim, "a", "corvette", "alpha", Vector3D(-2_000e3, 0, 0))
    _ship(sim, "d1", "destroyer", "beta", Vector3D(300e3, 0, 0))
    _ship(sim, "d2", "destroyer", "beta", Vector3D(300e3, 40e3, 0))
    for i in range(10):
        _round(sim, f"t{i:02d}", "d1", dv, Vector3D(0, (i - 5) * 300.0, 0),
               Vector3D(12_000, 0, 0), profile=profile)
    return sim


class TestSurplusDiversion:

    def test_reserve_surplus_is_diverted_to_unmet_need(self):
        sim = _surplus_sim("reserve")
        _steps(sim, 1)
        ev = _rt(sim)
        assert ev and {e.data["reason"] for e in ev} == {"datalink_surplus"}
        assert {e.target_id for e in ev} == {"d2"}
        # d1 keeps what it needs, d2 gets what it needs (10 = 5 + 5)
        assert len(ev) == NEED_DD
        left_on_d1 = sum(1 for tf in sim.torpedoes if tf.torpedo.target_id == "d1")
        assert left_on_d1 == 10 - NEED_DD >= NEED_DD

    def test_sprint_rounds_are_never_diverted(self):
        sim = _surplus_sim("sprint")
        _steps(sim, 1)
        assert not _rt(sim)

    def test_diversion_respects_fuel(self):
        sim = _surplus_sim("reserve", dv=0.6)   # cannot swing 40 km abeam
        _steps(sim, 1)
        assert not _rt(sim)

    def test_committed_rounds_are_not_diverted(self):
        sim = _surplus_sim("reserve")
        for tf in sim.torpedoes:      # 8 s out: inside LINK_COMMITTED_S
            tf.torpedo.position = Vector3D(300e3 - 96e3, tf.torpedo.position.y, 0)
        _steps(sim, 1)
        assert not [e for e in _rt(sim) if e.data["reason"] == "datalink_surplus"]


# ---------------------------------------------------------------------------
# The shared estimate
# ---------------------------------------------------------------------------

class TestRoundsNeeded:

    def test_soak_plus_kill(self):
        sim = _sim()
        bc = _ship(sim, "bc", "battlecruiser", "beta", Vector3D(0, 0, 0), pd=True)
        _ship(sim, "f", "frigate", "beta", Vector3D(20e3, 0, 0), pd=True)
        need = L.rounds_needed(sim, bc, cross_kps=12.0)
        assert need.cover == 4                    # 2 own + 2 escort turrets
        assert need.soak == 4 // L.turrets_to_blind(12.0, FLEET)
        assert need.kill == L.KILL_ROUNDS_HEAVY
        assert need.total == need.soak + need.kill
        slow = L.rounds_needed(sim, bc, cross_kps=8.0, energy_gj=11.0)
        assert slow.soak >= need.soak and slow.kill == 12

    def test_llm_layers_use_the_engine_estimate(self):
        from src.llm import fleet_intel
        assert fleet_intel.kill_rounds is L.kill_rounds
        assert fleet_intel.kill_rounds_at is L.kill_rounds_at
