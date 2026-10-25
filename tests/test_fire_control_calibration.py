"""
P(hit) calibration against what the engine actually delivered, per target
maneuver class - so errors in opposite directions cannot cancel in a pool.

tests/fixtures/fire_control_shots.json holds resolved gun rounds
(scripts/fire_control_calibration.py fixture) from three kinds of source:

- 8 LLM battle recordings (visualizer recordings, including the 2026-10 flash
  mirror that exposed the problem: every dodger under a whole fleet's fire).
  They predate the recorded fire-control inputs, so their TargetMotion inputs
  are rebuilt from the event log (the missed jink from the target's execution
  state). Their rounds at HEADING / BURN targets are left out: those were
  fired before such targets were led by their drive's output.
- 32 seeded heuristic battles on the current engine (fresh_s<seed>_b<budget>:
  scripts/run_draft_battle.py --auto-draft --no-admirals --seed S --budget B,
  seeds 11 and 31-61, budgets 100-200), whose shots record the forecast's
  inputs ('fc').
- 1 in 20 rounds of the seeded live-engine harness (fire_control_calibration
  harness --runs 30 --classes padlock_dodge,hold_dodge,evasive,burn
  --shooters 1,2,4,8): 1, 2, 4 or 8 shooters against dodging (PADLOCK /
  HOLD_RANGE + evade 'dodge'), EVASIVE and HEADING-turning frigates..
  dreadnoughts, 40-400 km.

The model's P(hit) for a round is its conversion probability x the track
probability for that hull, TargetMotion and flight time - the same
displacement_sigma_m / track_probability calls the FIRE_WHEN_OPTIMAL gate
makes. Observed rates are valid targets only while the engine's dispersion
roll consumes the legacy conversion formula
(_resolve_projectile_hit_geometric); if that roll, the gate, evasion or the
lead changes, regenerate the fixture.

Buckets: maneuver class (EVASIVE, dodge = the evade overlay on another
maneuver, HEADING:turn / HEADING:burn, INTERCEPT, PADLOCK ...) x 100 km;
harness dodgers also split by one shooter vs a fleet of 2-8. A bucket with
n >= 30 and >= 5 hits must be within RATIO_TOLERANCE; one with fewer hits
must be Poisson-plausible both ways. The one exception is the known residual
of the LLM flash mirror's dodgers (firecontrol.JINK_ACCEL_FRACTION): pinned
on its own below, so it can neither grow back toward the old 10-40x nor be
silently absorbed.
"""

import collections
import importlib.util
import json
import math

import pytest

FIXTURE = "tests/fixtures/fire_control_shots.json"
RATIO_TOLERANCE = 1.5
MIN_BUCKET = 30
FLASH_MIRROR = "p4_new_flash_mirror_150pt"


def _fcc():
    spec = importlib.util.spec_from_file_location("fcc", "scripts/fire_control_calibration.py")
    fcc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fcc)
    return fcc


@pytest.fixture(scope="module")
def fcc():
    return _fcc()


@pytest.fixture(scope="module")
def shots(fcc):
    with open(FIXTURE) as f:
        data = json.load(f)
    hulls = fcc.HullTable()
    columns = data["columns"]
    rows = []
    for raw in data["rows"]:
        r = dict(zip(columns, raw))
        r["source"] = data["sources"][r["source"]]
        r["target_type"] = data["target_types"][r["target_type"]]
        r["class"] = data["classes"][r["class"]]
        r["harness"] = r["source"].startswith("harness:")
        r["new"] = hulls.predict(r)
        r["old"] = r["p_conv"]
        rows.append(r)
    return rows


def _bucket(r):
    cls = r["class"]
    if r["harness"] and cls in ("dodge", "EVASIVE"):
        cls += ":fleet" if r["n"] >= 2 else ":alone"
    return cls, min(int(r["range_km"] // 100), 5)


def _summary(group):
    n = len(group)
    hits = sum(r["hit"] for r in group)
    return n, hits, sum(r["new"] for r in group), sum(r["old"] for r in group)


def _poisson_ok(expected, hits):
    """Expected and observed hits agree within ~3 sigma (Poisson), both ways."""
    slack = 3.0 * math.sqrt(max(expected, hits, 1.0)) + 2.0
    return abs(expected - hits) <= slack


def _check(rows, label):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[_bucket(r)].append(r)
    report, failures, checked = [], [], set()
    for key in sorted(groups):
        n, hits, new, old = _summary(groups[key])
        line = (f"{label} {key[0]} {key[1] * 100}-{key[1] * 100 + 99} km: n={n} hits={hits} "
                f"forecast={new:.1f} old={old:.1f}")
        report.append(line)
        if n < MIN_BUCKET:
            continue
        checked.add(key)
        if hits >= 5:
            if not 1 / RATIO_TOLERANCE <= new / hits <= RATIO_TOLERANCE:
                failures.append(line)
        elif not _poisson_ok(new, hits):
            failures.append(line)
    assert not failures, "miscalibrated:\n" + "\n".join(failures) + "\n\nall:\n" + "\n".join(report)
    return checked


def test_battle_rounds_are_calibrated_per_maneuver_class(shots):
    checked = _check([r for r in shots if not r["harness"] and r["source"] != FLASH_MIRROR], "battle")
    classes = {k[0] for k in checked}
    # Every class the battles exercise is checked on its own.
    for cls in ("EVASIVE", "HEADING:turn", "HEADING:burn", "INTERCEPT", "PADLOCK"):
        assert cls in classes, (cls, sorted(classes))
    assert len(checked) >= 15


def test_harness_rounds_are_calibrated_per_maneuver_class(shots):
    checked = _check([r for r in shots if r["harness"]], "harness")
    classes = {k[0] for k in checked}
    for cls in ("dodge:alone", "dodge:fleet", "EVASIVE:alone", "EVASIVE:fleet", "HEADING:turn"):
        assert cls in classes, (cls, sorted(classes))


def test_the_flash_mirror_dodgers_are_the_known_residual(shots):
    """The motivating case: in the 2026-10 flash mirror every ship flew
    PADLOCK / HOLD_RANGE + evade 'dodge' under a whole fleet's fire. Not one
    of its 188 rounds at 100-300 km landed. The conversion number alone
    promised ~96 hits there and the single-shooter dodge model ~30; the
    forecast now says ~15. The rest is the residual named at
    firecontrol.JINK_ACCEL_FRACTION: those LLM-flown dodgers moved 2-4x
    farther during a flight than seeded engine dodgers under the same fire,
    which nothing visible at fire time predicts - the doctrine tells
    commanders instead (test_doctrine_states_the_residual). Pinned both ways:
    a forecast drifting back toward the old number fails, and so does one
    tuned to this single battle at the expense of the rest."""
    mirror = [r for r in shots if r["source"] == FLASH_MIRROR and r["class"] == "dodge"]
    mid = [r for r in mirror if 100 <= r["range_km"] < 300]
    n, hits, new, old = _summary(mid)
    assert (n, hits) == (188, 0)
    assert old > 80
    assert 8 < new < 20, new
    near = [r for r in mirror if r["range_km"] < 100]
    n, hits, new, _ = _summary(near)
    assert hits >= 5 and 1.0 < new / hits < 2.0, (n, hits, new)


def test_doctrine_states_the_residual(shots):
    """Captains and admirals are told the measured residual, in numbers
    bound to the fixture."""
    from src.llm.prompts import ADMIRAL_GUN_DOCTRINE, GUN_PHYSICS_REFERENCE
    mid = [r for r in shots if r["source"] == FLASH_MIRROR and r["class"] == "dodge"
           and 100 <= r["range_km"] < 300]
    n, hits, new, _ = _summary(mid)
    claim = f"{hits} of {n} rounds at 100-300km landed where this model promised ~{new:.0f}"
    for doctrine in (GUN_PHYSICS_REFERENCE, ADMIRAL_GUN_DOCTRINE):
        assert claim in " ".join(doctrine.split())


def test_mid_jink_dodgers_are_harder_and_the_forecast_says_so(shots):
    """Rounds that find a dodger already jinking (its drive burning across
    the line of fire, which its lead ignores) land less often than rounds at
    a dodger between jinks - fleet fire keeps a dodger jinking - and the
    forecast follows (JINK_ACCEL_FRACTION)."""
    def rate(jinking):
        g = [r for r in shots if r["harness"] and r["class"] == "dodge"
             and (r["jink"] > 0.5) == jinking and 100 <= r["range_km"] < 300]
        n, hits, new, _ = _summary(g)
        assert n >= 100
        return hits / n, new / n
    (obs_j, new_j), (obs_c, new_c) = rate(True), rate(False)
    assert obs_j < obs_c
    assert new_j < new_c


def test_the_old_number_was_the_problem(shots):
    """Why this exists: the conversion number alone promised 20-35% per round
    at dodging targets beyond 200 km, where ~2% landed."""
    far = [r for r in shots if not r["harness"] and r["class"] in ("dodge", "EVASIVE")
           and r["range_km"] >= 200]
    n, hits, new, old = _summary(far)
    assert n > 500
    assert old / n > 0.2 and old > 8 * hits
    assert new < 2 * hits + 15


def test_model_beats_the_old_number_on_likelihood(shots):
    def nll(key):
        return -sum(math.log(min(0.999, max(1e-4, r[key]))) if r["hit"]
                    else math.log(1 - min(0.999, max(1e-4, r[key]))) for r in shots)
    assert nll("new") < 0.85 * nll("old")


@pytest.mark.parametrize("seed, cls, hull, range_km, shooters, shooter_type", [
    (21, "padlock_dodge", "destroyer", 120, 1, "battleship"),
    (26, "padlock_dodge", "battleship", 300, 1, "battleship"),
    (31, "padlock_dodge", "battleship", 160, 4, "cruiser"),
    (17, "burn", "cruiser", 160, 1, "battleship"),
])
def test_live_engine_delivers_the_forecast(fcc, seed, cls, hull, range_km, shooters, shooter_type):
    """The fixture is a snapshot; this re-runs the engine
    (scripts/fire_control_calibration.py harness, seeded): a battleship, or a
    fleet of four cruisers, fires FIRE_IMMEDIATE for 150 s at a dodging hull
    (PADLOCK + evade 'dodge') or one turning onto a HEADING burn. The hits
    must be what the recorded per-round P(hit) forecast, within
    Poisson-binomial noise - and, at range or under fleet fire, far from what
    the conversion number alone promised. If evasion, the gate, the lead or
    the dispersion roll changes, this moves."""
    rounds = fcc.harness_run(seed, cls, hull, range_km, fcc.load_fleet(),
                             shooters=shooters, spread_deg=40.0, shooter_type=shooter_type)
    assert len(rounds) >= 40
    hits = sum(r["outcome"] == "hit" for r in rounds)
    expected = sum(r["p_shown"] for r in rounds)
    sd = math.sqrt(sum(r["p_shown"] * (1 - r["p_shown"]) for r in rounds))
    assert abs(hits - expected) <= 3 * sd + 2, (hits, expected, sd)
    if range_km >= 200 or shooters > 1:
        old = sum(r["p_conv"] for r in rounds)
        old_sd = math.sqrt(sum(r["p_conv"] * (1 - r["p_conv"]) for r in rounds))
        assert old - hits > 3 * old_sd + 2, (hits, old, old_sd)
