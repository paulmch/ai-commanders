"""
P(hit) calibration against what the engine actually delivered, per target
maneuver class - so errors in opposite directions cannot cancel in a pool.

tests/fixtures/fire_control_shots.json holds resolved gun rounds
(scripts/fire_control_calibration.py fixture, --free-keep 4 --harness-keep 1)
flown on the CURRENT engine, from two kinds of source:

- 64 seeded heuristic battles (fresh_<doctrine>_s<seed>_b<budget>:
  scripts/run_draft_battle.py --auto-draft --no-admirals --doctrine D --seed S
  --budget B, seeds 11 and 31-61, budgets 100-200 as listed in the fixture's
  sources), 32 with --doctrine legacy and 32 with --doctrine tactical (seed 58
  fires no gun in either), whose shots record the forecast's inputs ('fc').
  Their INTERCEPT / PADLOCK rounds closer than 300 km are kept 1 in 4.
- Every round of the seeded live-engine harness (fire_control_calibration
  harness --runs 30 --shooters 1,2,4,8, classes padlock_dodge, hold_dodge,
  evasive, burn, heading_dodge, intercept_dodge; the burn class only while the
  target is still turning onto its HEADING): 1, 2, 4 or 8 shooters against
  dodging (PADLOCK / HOLD_RANGE / a HEADING burn / an INTERCEPT + evade
  'dodge'; on a burning order the overlay
  keeps the burn and swings it - the 'swing' column, firecontrol.SWING_*),
  EVASIVE and HEADING-turning frigates..dreadnoughts, 40-400 km. All of them, not 1 in 20: thinned to 1 in 20 the 200-499 km
  buckets held 5-10 hits, where a 1.5x ratio is noise both ways.

tests/fixtures/fire_control_shots_historical.json is a separate, HISTORICAL
set: 8 battles recorded on the pre-pass-5 engine (scalar rotation, no weapon
arcs), including the 2026-10 LLM flash mirror. They cannot validate the
current engine and no calibration bucket reads them; only the flash-mirror
residual pin (and the doctrine line bound to it) does. Their TargetMotion
inputs are rebuilt from the event log, and their rounds at HEADING / BURN
targets are left out (fired before such targets were led by their drive).

The model's P(hit) for a round is its conversion probability x the track
probability for that hull, TargetMotion and flight time - the same
displacement_sigma_m / track_probability calls the FIRE_WHEN_OPTIMAL gate
makes. Observed rates are valid targets only while the engine's dispersion
roll consumes the legacy conversion formula
(_resolve_projectile_hit_geometric); if that roll, the gate, evasion or the
lead changes, regenerate the fixture.

Buckets: maneuver class (EVASIVE, dodge:<base> = the evade overlay on another
maneuver, by base maneuver - dodge:PADLOCK, dodge:HOLD_RANGE -, HEADING:turn /
HEADING:burn / HEADING:attitude, HOLD_RANGE,
INTERCEPT, PADLOCK ...) x 100 km; harness dodgers also split by one shooter
vs a fleet of 2-8. A bucket with n >= 30 and >= 5 hits must be within
RATIO_TOLERANCE; one with fewer hits must be Poisson-plausible both ways.
The exceptions are named in RESIDUALS and pinned on their own below, so they
can neither grow nor be silently absorbed.
"""

import collections
import importlib.util
import json
import math

import pytest

FIXTURE = "tests/fixtures/fire_control_shots.json"
HISTORICAL_FIXTURE = "tests/fixtures/fire_control_shots_historical.json"
RATIO_TOLERANCE = 1.5
MIN_BUCKET = 30
FLASH_MIRROR = "p4_new_flash_mirror_150pt"

# Buckets outside RATIO_TOLERANCE on the current engine, pinned on their own
# (test_known_residuals_stay_where_they_are): (label, class, 100 km band) ->
# allowed forecast/observed range.
RESIDUALS = {
    # A lone shooter at EVASIVE targets 200-299 km: its hits are on
    # dreadnoughts, in a few engagements; the forecast is 0.63x.
    ("harness", "EVASIVE:alone", 2): (0.5, 0.8),
    # Dodgers under a whole fleet's fire (2-8 shooters), even after the calm /
    # reacting split (firecontrol.DODGE_CALM_ACCEL_FRACTION): at 200-299 km
    # PADLOCK + dodge targets are hit 2.8x less than forecast and EVASIVE ones
    # 1.8x less; EVASIVE at 400-499 km 1.5x more. Whole engagements carry it
    # (test_the_fleet_fire_residual_is_carried_by_whole_engagements); the
    # doctrine tells commanders (prompts.GUN_PHYSICS_REFERENCE).
    ("harness", "dodge:PADLOCK:fleet", 2): (2.3, 3.4),
    ("harness", "EVASIVE:fleet", 2): (1.45, 2.1),
    ("harness", "EVASIVE:fleet", 4): (0.45, 0.85),
    # 5 hits where 1.4 were forecast: a lone shooter at a PADLOCK dodger 300-399
    # km (the dodge term over-charges the longest flights of a calm dodger).
    ("harness", "dodge:PADLOCK:alone", 3): (0.15, 0.45),
    # Battle rounds at long range: coasting PADLOCK targets at 400-499 km (0.67x)
    # and attitude-only HEADING targets at 300-399 km (0.62x) are hit more than
    # forecast - the unforeseen-maneuver term, fitted on order changes at
    # shorter range, over-charges the longest flights (neither a cap nor a
    # power law on it fitted better held out); HEADING burners at 300-399 km
    # less (1.66x, 5 hits of 32 rounds).
    # The evade overlay on a burning order (evadeburn round 2). Thin buckets
    # beyond 300 km swing with which few engagements land in them: on harness
    # seeds the fixture does not hold (runs 200-255 per shooter count, the
    # review's calib_fresh) the same buckets read HEADING alone 300-399
    # 1.16x, INTERCEPT alone 300-399 1.25x (4 hits), INTERCEPT fleet 400-499
    # 0.88x, HOLD_RANGE fleet 300-399 1.19x and 400-499 2.67x (4 hits). Each
    # band holds both seed sets. HOLD_RANGE dodgers under a fleet's fire
    # beyond 400 km are hit less than forecast in both (the dodge:PADLOCK
    # fleet residual's family: a range keeper jinks at full throttle between
    # its swung correction burns).
    ("harness", "dodge:HEADING:alone", 3): (0.5, 1.3),
    ("harness", "dodge:INTERCEPT:alone", 3): (1.1, 2.6),
    ("harness", "dodge:INTERCEPT:fleet", 4): (0.8, 1.8),
    ("harness", "dodge:HOLD_RANGE:fleet", 3): (1.1, 2.5),
    ("harness", "dodge:HOLD_RANGE:fleet", 4): (2.0, 3.6),
    ("battle", "PADLOCK", 4): (0.5, 0.8),
    ("battle", "HEADING:attitude", 3): (0.5, 0.8),
    ("battle", "HEADING:burn", 3): (1.3, 2.1),
}


def _fcc():
    spec = importlib.util.spec_from_file_location("fcc", "scripts/fire_control_calibration.py")
    fcc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fcc)
    return fcc


@pytest.fixture(scope="module")
def fcc():
    return _fcc()


def _load(fcc, path):
    with open(path) as f:
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


@pytest.fixture(scope="module")
def shots(fcc):
    return _load(fcc, FIXTURE)


@pytest.fixture(scope="module")
def historical(fcc):
    return _load(fcc, HISTORICAL_FIXTURE)


def _bucket(r):
    cls = r["class"]
    if r["harness"] and (cls.startswith("dodge") or cls == "EVASIVE"):
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
        if n < MIN_BUCKET or (label,) + key in RESIDUALS:
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
    battle = [r for r in shots if not r["harness"]]
    assert {r["source"].split("_")[1] for r in battle} == {"legacy", "tactical"}
    checked = _check(battle, "battle")
    classes = {k[0] for k in checked}
    # Every class the battles exercise is checked on its own.
    for cls in ("EVASIVE", "HEADING:turn", "HEADING:burn", "HEADING:attitude", "HOLD_RANGE",
                "INTERCEPT", "PADLOCK"):
        assert cls in classes, (cls, sorted(classes))
    assert len(checked) >= 15


def test_harness_rounds_are_calibrated_per_maneuver_class(shots):
    checked = _check([r for r in shots if r["harness"]], "harness")
    classes = {k[0] for k in checked}
    for cls in ("dodge:PADLOCK:alone", "dodge:PADLOCK:fleet", "dodge:HOLD_RANGE:alone",
                "dodge:HOLD_RANGE:fleet", "EVASIVE:alone", "EVASIVE:fleet", "HEADING:turn"):
        assert cls in classes, (cls, sorted(classes))


def test_known_residuals_stay_where_they_are(shots):
    groups = collections.defaultdict(list)
    for r in shots:
        groups[("harness" if r["harness"] else "battle",) + _bucket(r)].append(r)
    for key, (lo, hi) in RESIDUALS.items():
        n, hits, new, _ = _summary(groups[key])
        assert n >= MIN_BUCKET and hits >= 5, (key, n, hits)
        assert lo <= new / hits <= hi, (key, n, hits, new)


def test_the_flash_mirror_dodgers_are_the_known_residual(historical):
    """The motivating case: in the 2026-10 flash mirror every ship flew
    PADLOCK / HOLD_RANGE + evade 'dodge' under a whole fleet's fire. Not one
    of its 188 rounds at 100-300 km landed. The conversion number alone
    promised ~96 hits there and the single-shooter dodge model ~30; the
    forecast now says ~7. The rest is the residual named at
    firecontrol.JINK_ACCEL_FRACTION: those LLM-flown dodgers moved 2-4x
    farther during a flight than seeded engine dodgers under the same fire,
    which nothing visible at fire time predicts - the doctrine tells
    commanders instead (test_doctrine_states_the_residual). Pinned both ways:
    a forecast drifting back toward the old number fails, and so does one
    tuned to this single battle at the expense of the rest."""
    mirror = [r for r in historical if r["source"] == FLASH_MIRROR and r["class"].startswith("dodge")]
    mid = [r for r in mirror if 100 <= r["range_km"] < 300]
    n, hits, new, old = _summary(mid)
    assert (n, hits) == (188, 0)
    assert old > 80
    assert 5 < new < 15, new
    near = [r for r in mirror if r["range_km"] < 100]
    n, hits, new, _ = _summary(near)
    assert hits >= 5 and 1.0 < new / hits < 2.0, (n, hits, new)


def test_doctrine_states_the_residual(historical):
    """Captains and admirals are told the measured residual, in numbers
    bound to the fixture."""
    from src.llm.prompts import ADMIRAL_GUN_DOCTRINE, GUN_PHYSICS_REFERENCE
    mid = [r for r in historical if r["source"] == FLASH_MIRROR and r["class"].startswith("dodge")
           and 100 <= r["range_km"] < 300]
    n, hits, new, _ = _summary(mid)
    claim = f"{hits} of {n} rounds at 100-300km landed where this model promised ~{new:.0f}"
    for doctrine in (GUN_PHYSICS_REFERENCE, ADMIRAL_GUN_DOCTRINE):
        assert claim in " ".join(doctrine.split())
    # ... and the seeded fleet-fire residuals (RESIDUALS) they can expect.
    lo, hi = RESIDUALS[("harness", "dodge:PADLOCK:fleet", 2)]
    assert lo <= 2.8 <= hi
    lo, hi = RESIDUALS[("harness", "EVASIVE:fleet", 2)]
    assert lo <= 1.8 <= hi
    assert "nearly 3x less than shown (an EVADE-maneuver target 1.8x less)" in " ".join(
        GUN_PHYSICS_REFERENCE.split())
    assert "hit nearly 3x less than shown (EVADE maneuver 1.8x less)" in " ".join(
        ADMIRAL_GUN_DOCTRINE.split())


def test_mid_jink_dodgers_are_harder_and_the_forecast_says_so(shots):
    """PADLOCK + dodge targets at 100-300 km, by their state at the shot:
    a dodger caught mid-jink (its drive burning across the line of fire,
    which its lead ignores - JINK_ACCEL_FRACTION) is hit less often than one
    reacting but not burning across (TANK, BRACE ...), and a CALM one -
    flying its base maneuver, so this round provokes a dodge aimed at it
    (DODGE_CALM_ACCEL_FRACTION) - least of all; the forecast follows the
    same order (0.035 < 0.10 < 0.29 landed; 0.07 < 0.11 < 0.37 forecast)."""
    def rate(sel):
        g = [r for r in shots if r["harness"] and r["class"] == "dodge:PADLOCK"
             and sel(r) and 100 <= r["range_km"] < 300]
        n, hits, new, _ = _summary(g)
        assert n >= 500
        return hits / n, new / n
    calm = rate(lambda r: not r["react"])
    jinking = rate(lambda r: r["jink"] > 0.5)
    reacting = rate(lambda r: r["react"] and r["jink"] <= 0.5)
    assert calm[0] < jinking[0] < reacting[0]
    assert calm[1] < jinking[1] < reacting[1]


def test_the_old_number_was_the_problem(shots):
    """Why this exists: the conversion number alone promised 20-35% per round
    at dodging targets beyond 200 km, where ~2-4% lands (it was the LLM flash
    mirror that showed it; on the current engine the harness and battles
    say the same)."""
    far = [r for r in shots if (r["class"].startswith("dodge") or r["class"] == "EVASIVE")
           and r["range_km"] >= 200 and not r.get("swing")]
    n, hits, new, old = _summary(far)
    assert n > 5000
    assert old / n > 0.2 and old > 8 * hits
    assert new < 2 * hits + 15
    # A dodger whose evade overlay keeps an ordered burn and only swings it
    # (simulation module docstring) moves less: ~15% of these rounds land,
    # 2.5x fewer than the conversion number promised.
    swung = [r for r in shots if r["class"].startswith("dodge") and r["range_km"] >= 200
             and r.get("swing")]
    n, hits, new, old = _summary(swung)
    assert n > 5000
    assert old / n > 0.2 and old > 2.2 * hits
    assert new < 2 * hits + 15


def test_model_beats_the_old_number_on_likelihood(shots):
    def nll(key):
        return -sum(math.log(min(0.999, max(1e-4, r[key]))) if r["hit"]
                    else math.log(1 - min(0.999, max(1e-4, r[key]))) for r in shots)
    assert nll("new") < 0.85 * nll("old")


@pytest.mark.parametrize("seed, cls, hull, range_km, shooters, shooter_type, engagements", [
    (21, "padlock_dodge", "destroyer", 120, 1, "battleship", 1),
    (26, "padlock_dodge", "battleship", 300, 1, "battleship", 1),
    (31, "padlock_dodge", "battleship", 160, 4, "cruiser", 4),
    (17, "burn", "cruiser", 160, 1, "battleship", 1),
])
def test_live_engine_delivers_the_forecast(fcc, seed, cls, hull, range_km, shooters, shooter_type,
                                           engagements):
    """The fixture is a snapshot; this re-runs the engine
    (scripts/fire_control_calibration.py harness, seeded): a battleship, or a
    fleet of four cruisers, fires FIRE_IMMEDIATE for 150 s at a dodging hull
    (PADLOCK + evade 'dodge') or one turning onto a HEADING burn. The hits
    must be what the recorded per-round P(hit) forecast, within
    Poisson-binomial noise - and, at range or under fleet fire, far from what
    the conversion number alone promised. If evasion, the gate, the lead or
    the dispersion roll changes, this moves.

    The fleet-fire case pools four consecutive seeds: the rounds of ONE
    engagement are correlated (one dodger's path), so single engagements are
    over-dispersed against their own Poisson-binomial spread. Seeds 20-47 of
    that engagement: 708 hits against 791 forecast (1.12x), yet 5 of 28
    engagements fall outside 3 sigma, both ways (seed 31: 0 against 15; seed
    29: 45 against 28); on the scalar-rotation engine 3 of 28 did (974
    against 962) - committed jinks made it worse, the single-seed check was
    always fragile."""
    fleet = fcc.load_fleet()
    rounds = []
    for s in range(seed, seed + engagements):
        rounds += fcc.harness_run(s, cls, hull, range_km, fleet, shooters=shooters,
                                  spread_deg=40.0, shooter_type=shooter_type)
    assert len(rounds) >= 40
    hits = sum(r["outcome"] == "hit" for r in rounds)
    expected = sum(r["p_shown"] for r in rounds)
    sd = math.sqrt(sum(r["p_shown"] * (1 - r["p_shown"]) for r in rounds))
    assert abs(hits - expected) <= 3 * sd + 2, (hits, expected, sd)
    if range_km >= 200 or shooters > 1:
        old = sum(r["p_conv"] for r in rounds)
        old_sd = math.sqrt(sum(r["p_conv"] * (1 - r["p_conv"]) for r in rounds))
        assert old - hits > 3 * old_sd + 2, (hits, old, old_sd)



def test_the_fleet_fire_residual_is_carried_by_whole_engagements(shots):
    """Reviewer repro by_seed.py: PADLOCK + dodge targets under fleet fire at
    200-399 km, grouped by engagement (seed, hull, shooter count): of the 12
    engagements forecast 2+ hits, 10 got under half and none over double
    (51 hits vs 92 forecast; cluster-robust z -2.1). Rounds of one engagement
    share one dodger's path, so the residual is per engagement, not per
    round - which is why RESIDUALS holds it rather than a per-round term."""
    g = collections.defaultdict(lambda: [0, 0.0])
    for r in shots:
        if (r["harness"] and r["class"] == "dodge:PADLOCK" and r["n"] >= 2
                and 200 <= r["range_km"] < 400):
            k = (r["seed"], r["target_type"], r["n"])
            g[k][0] += r["hit"]
            g[k][1] += r["new"]
    big = [v for v in g.values() if v[1] >= 2.0]
    assert len(g) >= 30 and len(big) >= 8
    under = sum(1 for h, f in big if h < 0.5 * f)
    over = sum(1 for h, f in big if h > 2.0 * f)
    assert under >= 0.6 * len(big) and over <= 1, (len(big), under, over)
    d = [h - f for h, f in g.values()]
    z = sum(d) / math.sqrt(sum(x * x for x in d))
    assert z < -1.5, z
