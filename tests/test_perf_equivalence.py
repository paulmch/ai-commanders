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
