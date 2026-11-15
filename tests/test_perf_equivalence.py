"""
Engine optimisations must not change a single bit of any result.

Each optimised hot path is compared here, bit for bit, with a reference
implementation written the slow, obvious way (the code it replaced). The
end-to-end proof is ``scripts/perf_check.py`` (golden hashes of whole
battles); these tests localise a break to the function that caused it and
exercise inputs the golden battles may not reach.

For every fast path that depends on mutable state there is a test that
mutates that state and checks the fast path follows it.
"""

import json
import math
import random
import struct
from pathlib import Path

import pytest

from src.physics import Vector3D
from src.simulation import CombatSimulation, create_ship_from_fleet_data

FLEET_PATH = Path(__file__).resolve().parent.parent / "data" / "fleet_ships.json"


def load_fleet() -> dict:
    return json.loads(FLEET_PATH.read_text())


def bits(x: float) -> bytes:
    """Exact bit pattern of a float (distinguishes -0.0 from 0.0, NaNs)."""
    return struct.pack("<d", float(x))


def vbits(v: Vector3D) -> tuple:
    return (bits(v.x), bits(v.y), bits(v.z), type(v.x), type(v.y), type(v.z))


def rand_vec(rng: random.Random, scale: float) -> Vector3D:
    return Vector3D(rng.uniform(-scale, scale), rng.uniform(-scale, scale),
                    rng.uniform(-scale, scale))


# ---------------------------------------------------------------------------
# Vector3D
# ---------------------------------------------------------------------------

class TestVector3DInlinedOps:
    """normalized / distance_to / __rmul__ spell out what they used to chain."""

    def test_normalized_matches_division_by_magnitude(self):
        rng = random.Random(1)
        for _ in range(5000):
            v = rand_vec(rng, 10 ** rng.uniform(-6, 9))
            assert vbits(v.normalized()) == vbits(v / v.magnitude)

    def test_normalized_zero_vector_is_integer_zero(self):
        # The zero result has int components; that reaches JSON as 0, not 0.0.
        n = Vector3D(0.0, 0.0, 0.0).normalized()
        assert (n.x, n.y, n.z) == (0, 0, 0)
        assert all(type(c) is int for c in (n.x, n.y, n.z))

    def test_distance_to_matches_subtract_then_magnitude(self):
        rng = random.Random(2)
        for _ in range(5000):
            a, b = rand_vec(rng, 1e7), rand_vec(rng, 1e7)
            assert bits(a.distance_to(b)) == bits((a - b).magnitude)

    def test_rmul_matches_mul(self):
        rng = random.Random(3)
        for _ in range(1000):
            v, k = rand_vec(rng, 1e5), rng.uniform(-1e3, 1e3)
            assert vbits(k * v) == vbits(v * k)

    def test_slots_keep_copy_pickle_and_mutation(self):
        import copy
        import pickle
        v = Vector3D(1.5, -2.5, 3.5)
        assert not hasattr(v, "__dict__")
        for clone in (copy.copy(v), copy.deepcopy(v), pickle.loads(pickle.dumps(v))):
            assert vbits(clone) == vbits(v) and clone is not v
        # Components stay assignable and nothing derived is cached on the
        # instance: a mutated vector reports its new length.
        v.x = 10.0
        assert v.magnitude == math.sqrt(10.0 ** 2 + 2.5 ** 2 + 3.5 ** 2)
        assert vbits(v.normalized()) == vbits(v / v.magnitude)


# ---------------------------------------------------------------------------
# Closest approach
# ---------------------------------------------------------------------------

def reference_tca(proj_pos, proj_vel, target_pos, target_vel):
    """_calculate_time_to_closest_approach as it was written with Vector3D."""
    rel_pos = proj_pos - target_pos
    rel_vel = proj_vel - target_vel
    vel_mag_sq = rel_vel.dot(rel_vel)
    if vel_mag_sq < 1e-10:
        return 0.0, rel_pos.magnitude
    tca = -rel_pos.dot(rel_vel) / vel_mag_sq
    if tca < 0:
        return tca, rel_pos.magnitude
    closest_pos = rel_pos + rel_vel * tca
    return tca, closest_pos.magnitude


class TestClosestApproachScalar:
    def test_bit_identical_to_vector_form(self):
        sim = CombatSimulation(seed=1)
        rng = random.Random(4)
        branches = set()
        for i in range(20000):
            p, t = rand_vec(rng, 1e6), rand_vec(rng, 1e6)
            pv, tv = rand_vec(rng, 1e4), rand_vec(rng, 2e3)
            if i % 50 == 0:
                tv = Vector3D(pv.x, pv.y, pv.z)          # no relative motion
            if i % 7 == 0:
                # Aim the round at the target so both signs of TCA occur.
                d = (t - p).normalized() * rng.choice([-8000.0, 8000.0])
                pv = tv + d
            got = sim._calculate_time_to_closest_approach(p, pv, t, tv)
            want = reference_tca(p, pv, t, tv)
            assert (bits(got[0]), bits(got[1])) == (bits(want[0]), bits(want[1]))
            branches.add("static" if (pv - tv).dot(pv - tv) < 1e-10
                         else "past" if want[0] < 0 else "ahead")
        assert branches == {"static", "past", "ahead"}


# ---------------------------------------------------------------------------
# Swept hull check: bounding-sphere rejection
# ---------------------------------------------------------------------------

def reference_moving_cylinder(sim, start, end, ship, target_start, target_end, tolerance):
    """_check_moving_cylinder_intersection without the early rejection."""
    displacement = target_end - target_start
    hit, _, fraction = sim._check_line_cylinder_intersection_at_pos(
        start, end - displacement, ship, target_start, tolerance)
    if not hit:
        return False, None, None
    return (True, start + (end - start) * fraction,
            target_start + displacement * fraction)


def _hull(ship_type="destroyer", forward=None):
    return create_ship_from_fleet_data(
        "target", ship_type, "beta", load_fleet(), position=Vector3D(0, 0, 0),
        forward=forward or Vector3D(1, 0, 0))


def _same_sweep(a, b) -> bool:
    if a[0] != b[0]:
        return False
    if not a[0]:
        return a == b
    return vbits(a[1]) == vbits(b[1]) and vbits(a[2]) == vbits(b[2])


class TestMovingCylinderRejection:
    def _segments(self, rng, ship, n):
        """Segments at every distance scale around the hull, many grazing it."""
        reach = math.hypot(ship.geometry.length_m / 2.0, ship.geometry.radius_m + 500.0)
        for _ in range(n):
            target_start = rand_vec(rng, 1e5)
            target_end = target_start + rand_vec(rng, 30.0)
            near = target_start + rand_vec(rng, 1.0).normalized() * (
                reach * rng.choice([0.2, 0.9, 1.0, 1.1, 1.5, 2.0, 2.1, 3.0, 10.0, 500.0]))
            length = rng.choice([1.0, 8.0, 50.0, 800.0, 9000.0])
            direction = rand_vec(rng, 1.0).normalized()
            if rng.random() < 0.5:
                direction = (target_start - near).normalized()
            yield near, near + direction * length, target_start, target_end

    @pytest.mark.parametrize("ship_type", ["frigate", "destroyer", "dreadnought"])
    def test_same_answer_as_exact_check(self, ship_type):
        sim = CombatSimulation(seed=1)
        rng = random.Random(5)
        ship = _hull(ship_type, forward=rand_vec(rng, 1.0).normalized())
        hits = rejected = 0
        for start, end, ts, te in self._segments(rng, ship, 6000):
            got = sim._check_moving_cylinder_intersection(start, end, ship, ts, te, 500.0)
            want = reference_moving_cylinder(sim, start, end, ship, ts, te, 500.0)
            assert _same_sweep(got, want)
            hits += got[0]
        assert hits > 200          # the comparison is not vacuous

    def test_follows_geometry_and_tolerance_changes(self):
        """The rejection reads hull size and tolerance live (nothing cached):
        a segment rejected for a small hull is a hit once the hull is big
        enough, and vice versa."""
        sim = CombatSimulation(seed=1)
        ship = _hull("frigate")
        ts = te = Vector3D(0.0, 0.0, 0.0)
        start, end = Vector3D(0.0, 5000.0, 4000.0), Vector3D(0.0, -5000.0, 4000.0)
        assert sim._check_moving_cylinder_intersection(start, end, ship, ts, te, 0.0)[0] is False
        assert sim._check_moving_cylinder_intersection(start, end, ship, ts, te, 4500.0)[0] is True
        ship.geometry.radius_m = 4200.0
        got = sim._check_moving_cylinder_intersection(start, end, ship, ts, te, 0.0)
        assert got[0] is True
        assert _same_sweep(got, reference_moving_cylinder(sim, start, end, ship, ts, te, 0.0))
        ship.geometry.radius_m = 10.0
        assert sim._check_moving_cylinder_intersection(start, end, ship, ts, te, 0.0)[0] is False

    def test_no_geometry_uses_sphere_fallback(self):
        sim = CombatSimulation(seed=1)
        ship = _hull("frigate")
        ship.geometry = None
        ts = te = Vector3D(0.0, 0.0, 0.0)
        start, end = Vector3D(-1000.0, 10.0, 0.0), Vector3D(1000.0, 10.0, 0.0)
        got = sim._check_moving_cylinder_intersection(start, end, ship, ts, te, 0.0)
        assert _same_sweep(got, reference_moving_cylinder(sim, start, end, ship, ts, te, 0.0))
        assert got[0] is True


# ---------------------------------------------------------------------------
# Terminal micro-step fast path
# ---------------------------------------------------------------------------

def _no_fast_path(proj_flight, target_ship, target_start, time_remaining,
                  current_dist, micro_dt_max, max_micro_steps):
    """_coast_micro_steps that takes no steps: the full loop does everything."""
    return time_remaining, 0.0, 0, current_dist


def _gun_engagement(seed, target_type, range_km, target_maneuver, duration_s, fast,
                    mutate=None):
    """A battleship shooting every gun at one target. Returns the complete
    event stream and the full-precision end state."""
    import contextlib
    import io
    from src.firecontrol import WeaponsCommand, WeaponsOrder
    from src.simulation import Maneuver, ManeuverType

    fleet = load_fleet()
    rng = random.Random(seed)
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=seed)
    calls = {"n": 0, "steps": 0}
    if fast:
        real = sim._coast_micro_steps

        def counted(*args):
            out = real(*args)
            calls["n"] += 1
            calls["steps"] += out[2]
            return out
        sim._coast_micro_steps = counted
    else:
        sim._coast_micro_steps = _no_fast_path
    target = create_ship_from_fleet_data(
        "target", target_type, "beta", fleet, position=Vector3D(range_km * 1000.0, 0, 0),
        velocity=rand_vec(rng, 900.0), forward=rand_vec(rng, 1.0).normalized())
    shooter = create_ship_from_fleet_data(
        "shooter", "battleship", "alpha", fleet, position=Vector3D(0, 0, 0),
        forward=Vector3D(1, 0, 0))
    sim.add_ship(target)
    sim.add_ship(shooter)
    shooter.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, target_id="target")
    shooter.weapons_orders = {
        slot: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, slot, "target", min_hit_probability=0.0)
        for slot in shooter.weapons if not slot.startswith("pd_")}
    if target_maneuver == "coast":
        target.current_maneuver = Maneuver(ManeuverType.MAINTAIN, 0, duration=1e9, throttle=0.0)
    elif target_maneuver == "evasive":
        target.current_maneuver = Maneuver(ManeuverType.EVASIVE, 0, duration=1e9, throttle=1.0)
    else:
        target.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, throttle=1.0,
                                           target_id="shooter", evade="dodge")
    events = []
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.current_time < duration_s and not target.is_destroyed:
            if mutate is not None:
                mutate(sim, target)
            for ev in sim.step():
                events.append((ev.event_type.name, repr(ev.timestamp), ev.ship_id, ev.target_id,
                               json.dumps(ev.data, sort_keys=True, default=repr)))
    end = [repr(sim.rng.getstate()), repr(sim._damage_rng.getstate())]
    for ship in sim.ships.values():
        end += [repr(c) for v in (ship.position, ship.velocity, ship.forward)
                for c in (v.x, v.y, v.z)]
    for flight in sim.projectiles:
        p = flight.projectile.position
        end += [flight.projectile_id, repr(p.x), repr(p.y), repr(p.z),
                repr(flight.min_distance_to_target), repr(flight.prev_distance_to_target)]
    return events, end, calls


class TestMicroStepFastPath:
    """_coast_micro_steps runs the uneventful terminal micro-steps in scalar
    arithmetic; the battle must be bit-identical to running every micro-step
    through the full loop."""

    @pytest.mark.parametrize("seed,target_type,range_km,maneuver", [
        (11, "destroyer", 40, "coast"),        # hits (hull and proximity)
        (12, "frigate", 120, "evasive"),       # mostly misses
        (13, "cruiser", 80, "dodge"),          # evade overlay, PD ablation
        (14, "dreadnought", 60, "coast"),
    ])
    def test_engagement_bit_identical_to_full_loop(self, seed, target_type, range_km, maneuver):
        fast = _gun_engagement(seed, target_type, range_km, maneuver, 45.0, fast=True)
        slow = _gun_engagement(seed, target_type, range_km, maneuver, 45.0, fast=False)
        assert fast[0] == slow[0]
        assert fast[1] == slow[1]
        kinds = {e[0] for e in fast[0]}
        assert "PROJECTILE_IMPACT" in kinds or "PROJECTILE_MISS" in kinds
        # ... and the fast path really carried the terminal phase.
        assert fast[2]["n"] > 0 and fast[2]["steps"] > 500 * fast[2]["n"] / 2

    def test_reads_live_state_every_call(self):
        """Nothing is cached between calls: the target's velocity, attitude
        and hull size change mid-battle (burns, turns, and here a hull that
        is resized between ticks) and the fast path follows, still matching
        the full loop bit for bit."""
        def mutate(sim, target):
            if 18.0 <= sim.current_time < 30.0:
                target.geometry.radius_m += 15.0
                target.geometry.length_m += 40.0
                target.kinematic_state.velocity = (
                    target.kinematic_state.velocity + Vector3D(3.0, -7.0, 2.0))

        fast = _gun_engagement(21, "destroyer", 50, "coast", 45.0, fast=True, mutate=mutate)
        slow = _gun_engagement(21, "destroyer", 50, "coast", 45.0, fast=False, mutate=mutate)
        plain = _gun_engagement(21, "destroyer", 50, "coast", 45.0, fast=True)
        assert fast[0] == slow[0] and fast[1] == slow[1]
        assert fast[0] != plain[0]             # the mutation mattered

    def test_declines_what_it_does_not_model(self):
        """Unknown projectile types, hulls without geometry and rounds with
        no relative motion are left entirely to the full loop."""
        from src.simulation import ProjectileInFlight
        from src.projectile import KineticProjectile

        sim = CombatSimulation(seed=1)
        ship = _hull("destroyer")
        ship.kinematic_state.velocity = Vector3D(100.0, 0.0, 0.0)
        start = Vector3D(0.0, 0.0, 0.0)

        def flight(velocity, cls=KineticProjectile):
            proj = cls(position=Vector3D(-20000.0, 30.0, 0.0), velocity=velocity, mass_kg=10.0)
            return ProjectileInFlight("p1", proj, "shooter", "target")

        class OddRound(KineticProjectile):
            pass

        untouched = (1.0, 0.0, 0, 20001.0)
        closing = Vector3D(8000.0, 0.0, 0.0)
        assert sim._coast_micro_steps(flight(closing), ship, start, 1.0, 20001.0, 0.001, 5000)[2] > 0
        f = flight(closing, OddRound)
        assert sim._coast_micro_steps(f, ship, start, 1.0, 20001.0, 0.001, 5000) == untouched
        assert vbits(f.projectile.position) == vbits(Vector3D(-20000.0, 30.0, 0.0))
        f = flight(Vector3D(100.0, 0.0, 0.0))            # same velocity as the target
        assert sim._coast_micro_steps(f, ship, start, 1.0, 20001.0, 0.001, 5000) == untouched
        ship.geometry = None
        assert sim._coast_micro_steps(flight(closing), ship, start, 1.0, 20001.0, 0.001, 5000) == untouched


# ---------------------------------------------------------------------------
# Point-defense geometry helpers
# ---------------------------------------------------------------------------

def reference_threatens_ship(threat_pos, threat_vel, ship_pos, ship_vel=None,
                             threat_range_m=500_000):
    distance = threat_pos.distance_to(ship_pos)
    if distance > threat_range_m:
        return False
    rel_vel = threat_vel - ship_vel if ship_vel is not None else threat_vel
    to_target = ship_pos - threat_pos
    return rel_vel.dot(to_target) > 0


def reference_on_collision_course(obj_pos, obj_vel, target_pos, target_vel,
                                  miss_threshold_m=5000.0):
    rel_pos = obj_pos - target_pos
    rel_vel = obj_vel - target_vel
    rel_speed_sq = rel_vel.dot(rel_vel)
    if rel_speed_sq < 1e-10:
        return rel_pos.magnitude < miss_threshold_m
    t_cpa = -rel_pos.dot(rel_vel) / rel_speed_sq
    if t_cpa < 0:
        return False
    cpa_pos = rel_pos + rel_vel * t_cpa
    return cpa_pos.magnitude < miss_threshold_m


def reference_closing_speed(obj_pos, obj_vel, target_pos, target_vel):
    rel_pos = target_pos - obj_pos
    rel_vel = obj_vel - target_vel
    if rel_pos.magnitude < 1.0:
        return 0.0
    return rel_vel.dot(rel_pos.normalized())


class TestPdGeometryScalar:
    def _cases(self, seed, n):
        rng = random.Random(seed)
        for i in range(n):
            scale = rng.choice([0.3, 2e3, 6e3, 3e5, 6e5, 2e6])
            target = rand_vec(rng, 1e6)
            obj = target + rand_vec(rng, scale)
            tv = rand_vec(rng, 3e4)
            ov = tv + rand_vec(rng, rng.choice([1e-7, 50.0, 9e3]))
            if i % 3 == 0:      # aimed at (or directly away from) the target
                ov = tv + (target - obj).normalized() * rng.choice([-7e3, 7e3])
                ov = ov + rand_vec(rng, rng.choice([0.0, 5.0, 200.0]))
            yield obj, ov, target, tv

    def test_threatens_ship(self):
        sim = CombatSimulation(seed=1)
        seen = set()
        for obj, ov, target, tv in self._cases(6, 20000):
            for ship_vel in (tv, None):
                got = sim._threatens_ship(obj, ov, target, ship_vel)
                assert got is reference_threatens_ship(obj, ov, target, ship_vel)
                seen.add(got)
        assert sim._threatens_ship(Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(900.0, 0, 0),
                                   None, threat_range_m=800.0) is False
        assert seen == {True, False}

    def test_on_collision_course(self):
        sim = CombatSimulation(seed=1)
        seen = set()
        for obj, ov, target, tv in self._cases(7, 20000):
            got = sim._on_collision_course(obj, ov, target, tv)
            assert got is reference_on_collision_course(obj, ov, target, tv)
            seen.add(got)
        assert seen == {True, False}

    def test_closing_speed(self):
        sim = CombatSimulation(seed=1)
        zero = False
        for obj, ov, target, tv in self._cases(8, 20000):
            got = sim._calculate_closing_speed(obj, ov, target, tv)
            assert bits(got) == bits(reference_closing_speed(obj, ov, target, tv))
            zero = zero or (target - obj).magnitude < 1.0
        assert zero


class TestAngleToInlined:
    def test_bit_identical_to_chained_form(self):
        def reference(a, b):
            dot = a.dot(b)
            mags = a.magnitude * b.magnitude
            if mags == 0:
                return 0.0
            return math.acos(max(-1.0, min(1.0, dot / mags)))

        rng = random.Random(9)
        for i in range(20000):
            a = rand_vec(rng, 10 ** rng.uniform(-3, 7))
            b = rand_vec(rng, 10 ** rng.uniform(-3, 7))
            if i % 4 == 0:
                b = a * rng.choice([1.0, -1.0, 3.7, -1e-3])      # (anti)parallel: clamp
            if i % 97 == 0:
                b = Vector3D(0.0, 0.0, 0.0)
            assert bits(a.angle_to(b)) == bits(reference(a, b))
        nan = Vector3D(float("nan"), 1.0, 0.0)
        assert bits(nan.angle_to(Vector3D(1.0, 0.0, 0.0))) == bits(
            reference(nan, Vector3D(1.0, 0.0, 0.0)))


# ---------------------------------------------------------------------------
# PRESENT scoring: memoised armor trial
# ---------------------------------------------------------------------------

def reference_trial(section, ke_gj):
    """What _presentation_score did inline: trial the impact on a deep copy."""
    import copy
    trial = copy.deepcopy(section)
    return trial.apply_energy_damage(energy_gj=ke_gj, flat_chipping=0.3, impact_area_m2=0.1)[1]


class TestPresentationTrialMemo:
    def _section(self, ship, name="lateral"):
        return ship.armor.get_section(name)

    def test_matches_direct_trial_and_leaves_section_untouched(self):
        sim = CombatSimulation(seed=1)
        rng = random.Random(10)
        for ship_type in ("corvette", "destroyer", "battleship"):
            ship = _hull(ship_type)
            for name in ("nose", "lateral", "tail"):
                section = self._section(ship, name)
                before = dict(vars(section))
                for _ in range(40):
                    ke = rng.choice([0.0, 0.5, 3.0, 12.0, 80.0, 900.0]) * rng.uniform(0.5, 1.5)
                    for _repeat in range(2):          # second call is served from the memo
                        assert bits(sim._trial_energy_to_hull_gj(section, ke)) == bits(
                            reference_trial(section, ke))
                assert vars(section) == before
        assert sim._trial_hull_cache            # it did memoise

    @pytest.mark.parametrize("field,change", [
        ("thickness_cm", lambda v: v * 0.4),               # ablation
        ("thickness_cm", lambda v: 0.0),                   # breached
        ("chipping_fraction", lambda v: 0.35),
        ("kinetics_resist", lambda v: 0.1),
        ("chip_resist", lambda v: 0.9),
        ("baryonic_half_cm", lambda v: v * 3.0),
        ("heat_of_vaporization_mj_kg", lambda v: v * 0.25),
        ("density", lambda v: v * 0.5),
    ])
    def test_every_section_attribute_is_part_of_the_key(self, field, change):
        """Invalidation: a memoised answer is never served for armor that has
        changed - whatever attribute changed."""
        sim = CombatSimulation(seed=1)
        section = self._section(_hull("cruiser"))
        ke = 40.0
        first = sim._trial_energy_to_hull_gj(section, ke)
        entries = len(sim._trial_hull_cache)
        setattr(section, field, change(getattr(section, field)))
        after = sim._trial_energy_to_hull_gj(section, ke)
        assert len(sim._trial_hull_cache) == entries + 1          # a new key, not a hit
        assert bits(after) == bits(reference_trial(section, ke))
        assert bits(first) == bits(sim._trial_hull_cache[next(iter(sim._trial_hull_cache))])

    def test_real_damage_between_calls_is_seen(self):
        sim = CombatSimulation(seed=1)
        section = self._section(_hull("destroyer"))
        intact = sim._trial_energy_to_hull_gj(section, 60.0)
        for _ in range(6):                                   # real hits ablate the plate
            section.apply_energy_damage(energy_gj=60.0, flat_chipping=0.3, impact_area_m2=0.1)
            assert bits(sim._trial_energy_to_hull_gj(section, 60.0)) == bits(
                reference_trial(section, 60.0))
        assert sim._trial_energy_to_hull_gj(section, 60.0) > intact

    def test_memo_is_per_simulation_and_bounded(self):
        a, b = CombatSimulation(seed=1), CombatSimulation(seed=1)
        section = self._section(_hull("destroyer"))
        a._trial_energy_to_hull_gj(section, 5.0)
        assert "_trial_hull_cache" not in vars(b)
        a._TRIAL_HULL_CACHE_MAX = 16
        for i in range(200):
            assert bits(a._trial_energy_to_hull_gj(section, 1.0 + i)) == bits(
                reference_trial(section, 1.0 + i))
            assert len(a._trial_hull_cache) <= 16

    def test_unkeyable_section_is_trialled_directly(self):
        sim = CombatSimulation(seed=1)
        section = self._section(_hull("destroyer"))
        section.notes = ["unhashable"]                       # a list defeats the key
        assert bits(sim._trial_energy_to_hull_gj(section, 30.0)) == bits(
            reference_trial(section, 30.0))
        assert not sim._trial_hull_cache

    def test_presentation_score_unchanged_through_armor_damage(self):
        """_presentation_score on one simulation, before and after the armor
        is shot up, equals the same scoring done with fresh deep-copy trials
        (a second simulation whose memo is bypassed)."""
        sim, fresh = CombatSimulation(seed=1), CombatSimulation(seed=1)
        fresh._trial_energy_to_hull_gj = reference_trial
        ship = _hull("cruiser", forward=Vector3D(0.3, -0.8, 0.52).normalized())
        rng = random.Random(11)
        threats = []
        for _ in range(6):
            bearing = rand_vec(rng, 1.0).normalized()
            threats.append({"bearing": bearing, "ke_gj": rng.uniform(5.0, 120.0),
                            "rel_vel": bearing * -rng.uniform(4e3, 1.5e4) + rand_vec(rng, 800.0)})
        attitudes = [rand_vec(rng, 1.0).normalized() for _ in range(40)]
        for round_ in range(4):
            for forward in attitudes:
                assert sim._presentation_score(ship, forward, threats) == \
                    fresh._presentation_score(ship, forward, threats)
            for name in ("nose", "lateral", "tail"):
                self._section(ship, name).apply_energy_damage(
                    energy_gj=70.0 * (round_ + 1), flat_chipping=0.3, impact_area_m2=0.1)


# ---------------------------------------------------------------------------
# Recording serialisation
# ---------------------------------------------------------------------------

class TestRecordingToJsonWithoutDeepCopy:
    """BattleRecording.to_json serialises the fields directly instead of
    deep-copying the recording through asdict() first."""

    def _recording(self):
        from dataclasses import dataclass, field
        from src.llm.battle_recorder import BattleEvent, BattleRecording

        @dataclass
        class Nested:
            name: str
            values: list = field(default_factory=list)
            inner: object = None

        rec = BattleRecording()
        rec.battle_name = "to_json équivalence"
        rec.events = [
            BattleEvent(1.5, "shot_fired", "alpha_1", {"p": 0.1 + 0.2, "v": [1e-9, -0.0, 3]},
                        event_id="e1", sequence=1).to_dict(),
            {"event_id": "e2", "data": {"nested": Nested("n", [1, (2, 3)], Nested("deep"))},
             "tuple": (1, 2.5, None), "flag": True},
            BattleEvent(2.0, "raw_dataclass_event", None, {"k": {"x": float("inf")}}),
        ]
        rec.sim_trace = [{"t": float(i), "ships": {"a": {"pos": [i * 0.1, -i, 7.25e6]}}}
                         for i in range(5)]
        rec.provenance = {"seed": 7, "empty": {}, "list": []}
        return rec

    @pytest.mark.parametrize("indent", [2, None, 0])
    def test_same_text_as_asdict_route(self, indent):
        from dataclasses import asdict
        rec = self._recording()
        assert rec.to_json(indent=indent) == json.dumps(asdict(rec), indent=indent)

    def test_default_indent_and_to_dict_still_a_deep_copy(self):
        from dataclasses import asdict
        rec = self._recording()
        assert rec.to_json() == json.dumps(asdict(rec), indent=2)
        copy_ = rec.to_dict()
        copy_["events"][0]["data"]["p"] = "changed"
        assert rec.events[0]["data"]["p"] == 0.1 + 0.2

    def test_unserialisable_value_still_raises(self):
        rec = self._recording()
        rec.events.append({"data": {"bad": object()}})
        with pytest.raises(TypeError):
            rec.to_json()
