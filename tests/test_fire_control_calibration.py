"""
P(hit) calibration against what the engine actually delivered.

tests/fixtures/fire_control_shots.json holds 4.2k resolved gun rounds
(scripts/fire_control_calibration.py fixture) from 13 recorded battles (LLM
and heuristic fleets, including the 2026-10 flash mirrors that exposed the
problem) - every round at an evasion-armed target or 300+ km out, 1 in 4 of
the closer ones at other targets - plus 1 in 20 rounds from the seeded
engine harness against dodging/EVASIVE frigates..dreadnoughts. For each
round: range, time of flight, target hull, whether its evasion was armed
(and its throttle), the engine's conversion probability and whether it hit.

The model's P(hit) for a round is its conversion probability x the track
probability from CombatSimulation.target_motion for that hull - the same
calls the FIRE_WHEN_OPTIMAL gate makes. Observed rates are valid targets
only while the engine's dispersion roll consumes the legacy conversion
formula (_resolve_projectile_hit_geometric); if that roll changes,
regenerate the fixture.

Tolerances: within 1.5x for rounds at non-reacting targets (n >= 30, >= 5
hits). Rounds at dodging targets get 1.75x: how well a ship dodges depends
on how much fire it is under (one shooter in the harness, a fleet's worth
plus escort point defense in the flash mirror - that battle's 100-199 km
bucket sits at ~1.6x). The old number was 3-60x off there.
"""

import collections
import json
import math

import pytest

from src.firecontrol import HIT_TOLERANCE_M, displacement_sigma_m, track_probability
from src.simulation import CombatSimulation, Maneuver, ManeuverType, create_ship_from_fleet_data

FIXTURE = "tests/fixtures/fire_control_shots.json"
FREE_TOLERANCE = 1.5
ARMED_TOLERANCE = 1.75


@pytest.fixture(scope="module")
def shots():
    with open(FIXTURE) as f:
        data = json.load(f)
    with open("data/fleet_ships.json") as f:
        fleet = json.load(f)
    sim = CombatSimulation(seed=0)
    motion_cache = {}

    def track(target_type, armed, throttle, tof):
        key = (target_type, armed, throttle)
        if key not in motion_cache:
            ship = create_ship_from_fleet_data("t", target_type, "beta", fleet)
            if armed:
                ship.current_maneuver = Maneuver(
                    ManeuverType.PADLOCK, 0.0, throttle=1.0 if throttle is None else throttle,
                    target_id="x", evade="dodge")
            motion_cache[key] = (sim.target_motion(ship), HIT_TOLERANCE_M + ship.geometry.radius_m)
        motion, envelope = motion_cache[key]
        return track_probability(displacement_sigma_m(tof, motion), envelope)

    rows = []
    for src, rng_km, tof, ttype, armed, throttle, p_conv, hit in data["rows"]:
        source = data["sources"][src]
        target_type = data["target_types"][ttype]
        rows.append({
            "source": source, "harness": source.startswith("harness:"),
            "range_km": rng_km, "armed": bool(armed), "hit": hit, "old": p_conv,
            "new": min(0.99, p_conv * track(target_type, bool(armed), throttle, tof)),
        })
    return rows


def _buckets(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[("armed" if r["armed"] else "free", min(int(r["range_km"] // 100), 5))].append(r)
    return groups


def _plausible(expected_hits, hits):
    return expected_hits <= hits + 3.0 * math.sqrt(max(expected_hits, 1.0)) + 2.0


def _summary(group):
    n = len(group)
    hits = sum(r["hit"] for r in group)
    return n, hits, hits / n, sum(r["new"] for r in group) / n, sum(r["old"] for r in group) / n


def test_battle_rounds_are_calibrated(shots):
    checked, report = 0, []
    for (cls, rb), group in sorted(_buckets([r for r in shots if not r["harness"]]).items()):
        n, hits, obs, new, old = _summary(group)
        report.append(f"{cls} {rb * 100}-{rb * 100 + 99} km: n={n} obs={obs:.3f} new={new:.3f} old={old:.3f}")
        if n < 30:
            continue
        if hits >= 5:
            tol = ARMED_TOLERANCE if cls == "armed" else FREE_TOLERANCE
            assert 1 / tol <= new / obs <= tol, "\n".join(report)
            checked += 1
        else:
            # Next to nothing lands here: the forecast must say so (the old
            # number promised 20-30% per round, i.e. 50-70 hits in these
            # buckets) - no more expected hits than a Poisson 3-sigma allows.
            assert _plausible(n * new, hits), "\n".join(report)
    assert checked >= 8


def test_harness_rounds_at_dodging_targets_are_calibrated(shots):
    for (cls, rb), group in sorted(_buckets([r for r in shots if r["harness"]]).items()):
        n, hits, obs, new, _ = _summary(group)
        if n < 30:
            continue
        if hits >= 5:
            assert 1 / ARMED_TOLERANCE <= new / obs <= ARMED_TOLERANCE, (rb, n, obs, new)
        else:
            assert _plausible(n * new, hits), (rb, n, obs, new)


def test_the_low_hit_check_would_have_caught_the_old_number(shots):
    group = [r for r in shots if not r["harness"] and r["armed"] and 300 <= r["range_km"] < 400]
    n, hits, _, _, old = _summary(group)
    assert n >= 30 and hits < 5
    assert not _plausible(n * old, hits)


def test_the_old_number_was_the_problem(shots):
    """Why this exists: the conversion number alone promised 20-35% per round
    at dodging targets beyond 200 km, where ~1% landed."""
    far = [r for r in shots if not r["harness"] and r["armed"] and r["range_km"] >= 200]
    n, hits, obs, new, old = _summary(far)
    assert n > 300
    assert old > 0.2 and old > 8 * obs
    assert new < 3 * max(obs, 0.01)


def test_model_beats_the_old_number_on_likelihood(shots):
    def nll(key):
        return -sum(math.log(min(0.999, max(1e-4, r[key]))) if r["hit"]
                    else math.log(1 - min(0.999, max(1e-4, r[key]))) for r in shots)
    assert nll("new") < 0.85 * nll("old")
