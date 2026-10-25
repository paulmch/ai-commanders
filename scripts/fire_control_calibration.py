#!/usr/bin/env python3
"""
Fire-control calibration: is the per-shot P(hit) commanders see what the
engine actually delivers?

P(hit) = conversion x track (src/firecontrol.py). This script measures it
against what the engine does, in two ways, and writes the test fixture:

  recordings   Per-shot dataset from battle recordings (recording_version 3:
               shot_fired events with projectile_id, and their hit / miss
               events). Features: range, time of flight, target hull, the
               target's maneuver class at fire time (EVASIVE, a dodge-overlay
               maneuver, HEADING turning / burning / attitude-only,
               INTERCEPT, PADLOCK, ...), the conversion probability
               (recorded as conversion_probability; older recordings recorded
               only hit_probability, which WAS the conversion number), the
               forecast's TargetMotion inputs, outcome (hit / roll = lost the
               dispersion roll / gate = passed outside the 500 m gate).
               Recordings since the 2026-10 fleet-fire recalibration carry
               those inputs on every shot ('fc': missed jink burn, pending
               HEADING burn); for older ones they are
               reconstructed from the event log (approx=True): the jink from
               the target's execution state and maneuver, no pending burn.
               Every row also counts the ships that fired at the target in
               the last 30 s (n, a diagnostic: fleet fire vs one shooter).
  harness      Controlled engagements on the live engine: 1-8 shooters
               (battleship / cruiser / destroyer) firing FIRE_IMMEDIATE at one
               target that coasts, PADLOCKs, burns on a HEADING, flies
               EVASIVE, or PADLOCK / HOLD_RANGE with evade='dodge';
               frigate..dreadnought, 40-400 km. Seeded and deterministic.
  table        Calibration table: observed hit rate vs the recorded (old) and
               the current model's P(hit), per maneuver class x 100 km bucket.
  fixture      Compact shot list for tests/test_fire_control_calibration.py.

Examples:
    uv run python scripts/fire_control_calibration.py recordings \\
        visualizer/public/recordings/p4_*_flash_mirror_150pt.json --out shots.jsonl
    uv run python scripts/fire_control_calibration.py harness --runs 36 --shooters 1,2,4,8 \\
        --workers 14 --out harness.jsonl
    uv run python scripts/fire_control_calibration.py table shots.jsonl
    uv run python scripts/fire_control_calibration.py table harness.jsonl --by source
    uv run python scripts/fire_control_calibration.py fixture shots.jsonl harness.jsonl \\
        --out tests/fixtures/fire_control_shots.json

The conversion roll is the engine's own; observed hit rates in these
datasets are valid calibration targets only while that roll stays the legacy
conversion formula (see _resolve_projectile_hit_geometric).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.firecontrol import (  # noqa: E402
    HIT_TOLERANCE_M, TargetMotion, WeaponsCommand, WeaponsOrder,
    displacement_sigma_m, track_probability,
)
from src.physics import Vector3D  # noqa: E402
from src.simulation import (  # noqa: E402
    BALLISTIC_THREAT_HORIZON_S, EVASION_MIN_JINK_S, EVASION_OVERLAY_MODES,
    EVASION_TURN_ALLOWANCE_S, CombatSimulation, Maneuver, ManeuverType,
    SimulationEventType, create_ship_from_fleet_data,
)

FLEET_PATH = ROOT / "data" / "fleet_ships.json"
HARNESS_CLASSES = ("coast", "padlock", "burn", "evasive", "padlock_dodge", "hold_dodge")
HARNESS_HULLS = ("frigate", "destroyer", "cruiser", "battleship", "dreadnought")
HARNESS_RANGES_KM = (40, 80, 120, 160, 220, 300, 400)
HARNESS_SHOOTER_TYPES = ("battleship", "cruiser", "destroyer")
HARNESS_DURATION_S = 150.0
# Ships engaging a target (a diagnostic column): the shooter plus every other
# ship that fired at it this long before.
ENGAGED_WINDOW_S = 30.0


def load_fleet() -> dict:
    with open(FLEET_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Maneuver classes
# ---------------------------------------------------------------------------

def maneuver_class(row: dict) -> str:
    """
    The target's maneuver class at fire time, the calibration test's buckets:
    EVASIVE; 'dodge' (any other maneuver with the evade overlay); HEADING
    split into 'HEADING:turn' (its ordered burn not lit yet: turning onto
    the heading or a held start), 'HEADING:burn' and 'HEADING:attitude'
    (throttle 0); else the maneuver type (INTERCEPT, PADLOCK, HOLD_RANGE ...).
    """
    mt = row.get("mtype") or "NONE"
    if mt == "EVASIVE":
        return "EVASIVE"
    if row.get("dodge") is not None:
        return "dodge"
    if mt == "HEADING":
        if not (row.get("throttle") or 0.0) > 0.0:
            return "HEADING:attitude"
        return "HEADING:turn" if (row.get("pending_in") or row.get("turning")) else "HEADING:burn"
    return mt


# ---------------------------------------------------------------------------
# Model evaluation for a dataset row
# ---------------------------------------------------------------------------

class HullTable:
    """Per-class full-thrust acceleration and hull radius of a fresh ship."""

    def __init__(self, fleet: Optional[dict] = None):
        self.fleet = fleet or load_fleet()
        self._cache: Dict[str, tuple] = {}

    def hull(self, ship_type: str) -> tuple:
        if ship_type not in self._cache:
            ship = create_ship_from_fleet_data("x", ship_type, "alpha", self.fleet)
            self._cache[ship_type] = (
                ship.kinematic_state.max_acceleration_ms2(),
                ship.geometry.radius_m,
            )
        return self._cache[ship_type]

    def motion(self, row: dict) -> TargetMotion:
        """The TargetMotion the forecast used for a row (recorded 'fc'
        inputs, or the reconstruction for older recordings)."""
        accel, _ = self.hull(row["target_type"])
        dodge = row.get("dodge")
        return TargetMotion(
            accel_ms2=row.get("a") or accel,
            dodges=dodge is not None,
            evasive="EVASIVE" in (row.get("mtype"), row.get("class")),
            dodge_throttle=1.0 if dodge is None else max(0.0, min(1.0, dodge)),
            dodge_horizon_s=BALLISTIC_THREAT_HORIZON_S,
            dodge_min_time_s=EVASION_TURN_ALLOWANCE_S + EVASION_MIN_JINK_S,
            jink_accel_ms2=row.get("jink") or 0.0,
            pending_burn_ms2=row.get("pending") or 0.0,
            pending_burn_in_s=row.get("pending_in") or 0.0,
        )

    def predict(self, row: dict) -> float:
        """The current model's P(hit) for a recorded shot: its conversion
        probability x track for that hull and TargetMotion and flight time."""
        envelope = HIT_TOLERANCE_M + self.hull(row["target_type"])[1]
        sigma = displacement_sigma_m(row["tof_s"], self.motion(row))
        return min(0.99, row["p_conv"] * track_probability(sigma, envelope))


def _fc_features(fc: dict) -> dict:
    """Row features from a shot's recorded fire-control inputs."""
    return {"a": fc.get("a"), "dodge": fc.get("dodge"),
            "jink": fc.get("jink", 0.0),
            "pending": fc.get("pend", 0.0), "pending_in": fc.get("pend_in", 0.0),
            "approx": False}


def _lead_fraction(mtype: str, throttle: float, applied: float) -> float:
    """Fraction of full thrust, along the nose, that the engine's lead
    (_estimate_target_acceleration) assumes for this base maneuver - for
    reconstructing the missed jink burn of older recordings."""
    if mtype in ("HOLD_RANGE", "HEADING", "BURN"):
        return applied                  # reads the drive
    if mtype == "INTERCEPT":
        return throttle
    return 0.0                          # PADLOCK, MAINTAIN, ...


# ---------------------------------------------------------------------------
# recordings -> rows
# ---------------------------------------------------------------------------

def rows_from_recording(path: str, hulls: Optional[HullTable] = None) -> List[dict]:
    hulls = hulls or HullTable()
    with open(path) as f:
        rec = json.load(f)
    name = Path(path).stem
    ship_types = {}
    for side in ("alpha", "beta"):
        for ship in (rec.get(f"{side}_fleet") or {}).get("ships", []):
            ship_types[ship["ship_id"]] = ship.get("ship_type")
    events = rec.get("events") or []
    outcome: Dict[str, str] = {}
    for e in events:
        d = e.get("data") or {}
        pid = d.get("projectile_id")
        if not isinstance(pid, str) or not pid.startswith("proj"):
            continue
        if e["event_type"] == "hit":
            outcome[pid] = "hit"
        elif e["event_type"] == "miss" and pid not in outcome:
            det = d.get("detection")
            if det == "dispersion_roll":
                outcome[pid] = "roll"
            elif det in ("geometric", "coarse_tca"):
                outcome[pid] = "gate"
    # Per-ship timelines (by event sequence): orders and execution state.
    maneuvers = collections.defaultdict(list)
    execution = collections.defaultdict(list)
    for e in events:
        if e["event_type"] == "maneuver":
            maneuvers[e["ship_id"]].append((e["sequence"], e.get("data") or {}))
        elif e["event_type"] == "execution_state":
            execution[e["ship_id"]].append((e["sequence"], e.get("data") or {}))

    def at(timeline, seq):
        lo, hi = 0, len(timeline)
        while lo < hi:                  # last entry with sequence < seq
            mid = (lo + hi) // 2
            if timeline[mid][0] < seq:
                lo = mid + 1
            else:
                hi = mid
        i = lo - 1
        return timeline[i][1] if i >= 0 else {}

    shots = [e for e in events if e["event_type"] == "shot_fired"]
    fired_at = collections.defaultdict(list)     # target -> [(t, shooter)]
    for e in shots:
        fired_at[(e.get("data") or {}).get("target_id")].append((e["timestamp"], e["ship_id"]))
    rows = []
    for e in shots:
        d = e.get("data") or {}
        pid, target = d.get("projectile_id"), d.get("target_id")
        if pid not in outcome or ship_types.get(target) is None or not d.get("eta_s"):
            continue
        p = d.get("conversion_probability", d.get("hit_probability"))
        if p is None:
            continue
        t0, seq = e["timestamp"], e["sequence"]
        m = at(maneuvers.get(target, []), seq)
        mtype = m.get("maneuver_type") or "NONE"
        row = {
            "source": name, "t": t0, "range_km": d.get("distance_km"), "tof_s": d["eta_s"],
            "target_type": ship_types[target], "mtype": mtype, "throttle": m.get("throttle"),
            "p_conv": p, "p_shown": d.get("hit_probability"), "outcome": outcome[pid],
        }
        # Ships that fired at the target in the last ENGAGED_WINDOW_S, this
        # shooter included: a diagnostic (fleet fire vs one shooter), not a
        # model input.
        engaged = {s for t, s in fired_at[target] if t0 - ENGAGED_WINDOW_S <= t <= t0}
        engaged.add(e["ship_id"])
        row["n"] = len(engaged)
        if d.get("fc"):
            row.update(_fc_features(d["fc"]))
        else:
            x = at(execution.get(target, []), seq)
            dodges = mtype == "EVASIVE" or bool(m.get("evade"))
            throttle = 1.0 if m.get("throttle") is None else float(m.get("throttle"))
            accel = hulls.hull(ship_types[target])[0]
            jink = 0.0
            if dodges and mtype != "EVASIVE" and x.get("mode") in EVASION_OVERLAY_MODES:
                applied = float(x.get("applied_throttle") or 0.0)
                jink = accel * abs(applied - _lead_fraction(mtype, throttle, applied))
            row.update({"dodge": throttle if dodges else None,
                        "jink": jink, "pending": 0.0, "pending_in": 0.0, "approx": True})
            if mtype == "HEADING" and throttle > 0:
                # Turn phase (token throttle): classified, but its pending
                # burn cannot be rebuilt without the heading geometry.
                row["turning"] = float(x.get("applied_throttle") or 0.0) < 0.5 * throttle
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def _unit(rng: random.Random) -> Vector3D:
    while True:
        v = Vector3D(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
        if 0.1 < v.magnitude <= 1.0:
            return v.normalized()


def harness_run(seed: int, cls: str, target_type: str, range_km: float, fleet: dict,
                shooters: int = 1, spread_deg: float = 40.0,
                shooter_type: str = "battleship") -> List[dict]:
    """One controlled engagement - ``shooters`` ships of ``shooter_type``
    spread over ``spread_deg`` of bearing, PADLOCKed on one target and
    firing FIRE_IMMEDIATE - returns one row per resolved round."""
    rng = random.Random(seed)
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=seed)
    target = create_ship_from_fleet_data(
        "target", target_type, "beta", fleet, position=Vector3D(range_km * 1000.0, 0, 0),
        velocity=_unit(rng) * rng.uniform(0, 1500),
        forward=_unit(rng) if rng.random() < 0.5 else Vector3D(-1, 0, 0))
    sim.add_ship(target)
    for i in range(shooters):
        ang = (math.radians((i - (shooters - 1) / 2.0) * spread_deg / (shooters - 1))
               if shooters > 1 else 0.0)
        offset = Vector3D(-math.cos(ang), math.sin(ang), 0.0) * (range_km * 1000.0)
        position = target.position + offset
        if shooters > 1:
            position = position + Vector3D(0.0, 0.0, rng.uniform(-10e3, 10e3))
        shooter = create_ship_from_fleet_data(
            "shooter" if i == 0 else f"shooter{i}", shooter_type, "alpha", fleet,
            position=position, forward=(target.position - position).normalized())
        sim.add_ship(shooter)
        shooter.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, target_id="target")
        shooter.weapons_orders = {
            slot: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, slot, "target", min_hit_probability=0.0)
            for slot in shooter.weapons if not slot.startswith("pd_")}
    if cls == "coast":
        m = Maneuver(ManeuverType.MAINTAIN, 0, duration=1e9, throttle=0.0)
    elif cls == "burn":
        d = _unit(rng)
        m = Maneuver(ManeuverType.HEADING, 0, duration=1e9, throttle=rng.choice([0.3, 1.0]),
                     heading_direction={"x": d.x, "y": d.y, "z": d.z})
    elif cls == "evasive":
        m = Maneuver(ManeuverType.EVASIVE, 0, duration=1e9, throttle=1.0)
    else:
        kind = ManeuverType.HOLD_RANGE if cls == "hold_dodge" else ManeuverType.PADLOCK
        m = Maneuver(kind, 0, duration=1e9, throttle=1.0, target_id="shooter",
                     evade="dodge" if cls.endswith("_dodge") else None)
    target.current_maneuver = m
    shots: Dict[str, dict] = {}
    outcome: Dict[str, str] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.current_time < HARNESS_DURATION_S and not (target.is_destroyed or target.is_dying):
            for ev in sim.step():
                d = ev.data or {}
                pid = d.get("projectile_id")
                if ev.event_type == SimulationEventType.PROJECTILE_LAUNCHED:
                    shots[pid] = {"t": sim.current_time, "range_km": d.get("distance_km"),
                                  "tof_s": d.get("eta_s"),
                                  "p_conv": d.get("conversion_probability", d.get("hit_probability")),
                                  "p_shown": d.get("hit_probability"),
                                  **_fc_features(d.get("fc") or {})}
                elif pid in shots and pid not in outcome:
                    if ev.event_type == SimulationEventType.PROJECTILE_IMPACT:
                        outcome[pid] = "hit"
                    elif ev.event_type == SimulationEventType.PROJECTILE_MISS:
                        outcome[pid] = "roll" if d.get("detection") == "dispersion_roll" else "gate"
    tag = cls if shooters == 1 else f"{cls}:x{shooters}"
    return [dict(shot, source=f"harness:{tag}", target_type=target_type, n=shooters,
                 mtype=m.maneuver_type.name, throttle=m.throttle, outcome=outcome[pid], seed=seed)
            for pid, shot in shots.items() if pid in outcome and shot["tof_s"]]


def _harness_job(args) -> List[dict]:
    seed, cls, target_type, range_km, shooters, spread, shooter_type = args
    return harness_run(seed, cls, target_type, range_km, load_fleet(),
                       shooters=shooters, spread_deg=spread, shooter_type=shooter_type)


def harness_jobs(runs: int, classes: Iterable[str], shooters: Iterable[int] = (1,)) -> List[tuple]:
    rng = random.Random(12345)
    jobs = []
    for n in shooters:
        for i in range(runs):
            target_type = HARNESS_HULLS[i % len(HARNESS_HULLS)]
            range_km = rng.choice(HARNESS_RANGES_KM)
            spread = rng.choice((10.0, 40.0, 90.0))
            # One shooter: the battleship of the original single-shooter harness.
            shooter_type = rng.choice(HARNESS_SHOOTER_TYPES) if n > 1 else "battleship"
            for cls in classes:
                jobs.append((1000 + 1000 * (n - 1) + i, cls, target_type, range_km, n, spread,
                             shooter_type))
    return jobs


# ---------------------------------------------------------------------------
# table / fixture
# ---------------------------------------------------------------------------

def read_rows(paths: Iterable[str]) -> List[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def bucket_key(row: dict, by: str) -> tuple:
    rb = min(int((row["range_km"] or 0.0) // 100), 5)
    if by == "source":
        return (row["source"], maneuver_class(row))
    if by == "fleet":
        return (maneuver_class(row) + (":fleet" if (row.get("n") or 1) >= 2 else ":alone"), rb)
    return (maneuver_class(row), rb)


def calibration_table(rows: List[dict], hulls: HullTable, by: str = "range") -> List[dict]:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[bucket_key(r, by)].append(r)
    out = []
    for key in sorted(groups, key=str):
        g = groups[key]
        n = len(g)
        hits = sum(r["outcome"] == "hit" for r in g)
        out.append({
            "bucket": key, "n": n, "hits": hits, "observed": hits / n,
            "gate_pass": sum(r["outcome"] != "gate" for r in g) / n,
            "old": sum(r["p_conv"] for r in g) / n,
            "new": sum(hulls.predict(r) for r in g) / n,
        })
    return out


def _ratio(pred: float, obs: float) -> float:
    return (pred + 0.005) / (obs + 0.005)


def print_table(table: List[dict]) -> None:
    print(f"{'bucket':40s} {'n':>5s} {'hits':>5s} {'obs':>6s} {'gate':>5s} "
          f"{'old P':>6s} {'old/obs':>7s} {'new P':>6s} {'new/obs':>7s}")
    for t in table:
        b = t["bucket"]
        label = (f"{b[0]} {b[1] * 100}-{b[1] * 100 + 99} km" if isinstance(b[1], int) else f"{b[0]} {b[1]}")
        print(f"{label:40s} {t['n']:5d} {t['hits']:5d} {t['observed']:6.3f} {t['gate_pass']:5.2f} "
              f"{t['old']:6.3f} {_ratio(t['old'], t['observed']):6.2f}x {t['new']:6.3f} "
              f"{_ratio(t['new'], t['observed']):6.2f}x")


FIXTURE_COLUMNS = ["source", "range_km", "tof_s", "target_type", "class", "dodge", "n",
                   "jink", "pending", "pending_in", "p_conv", "hit"]


def write_fixture(rows: List[dict], out: str, free_keep: int, harness_keep: int,
                  pre_lead_fix: Iterable[str] = ()) -> None:
    """Every battle shot except the plentiful INTERCEPT / PADLOCK ones
    closer than 300 km, of which an evenly spaced 1 in free_keep per source,
    class and 100 km bucket; harness rounds 1 in harness_keep.
    Harness rounds at targets that neither dodge nor turn onto a HEADING are
    left out: their orders never change, which no battle reproduces. Rounds
    at HEADING / BURN targets (dodging or not) from ``pre_lead_fix`` sources
    are left out (recorded before those targets were led by their drive's
    output)."""
    kept = []
    counters = collections.Counter()
    pre = set(pre_lead_fix)
    for r in rows:
        harness = r["source"].startswith("harness:")
        cls = maneuver_class(r)
        if harness and r.get("dodge") is None and cls != "HEADING:turn":
            continue
        if r["source"] in pre and r.get("mtype") in ("HEADING", "BURN"):
            continue
        key = (r["source"], cls, min(int((r["range_km"] or 0.0) // 100), 5))
        counters[key] += 1
        far = (r["range_km"] or 0.0) >= 300.0
        common = cls in ("INTERCEPT", "PADLOCK")
        keep_every = harness_keep if harness else (free_keep if common and not far else 1)
        if (counters[key] - 1) % keep_every == 0:
            kept.append(r)
    sources = sorted({r["source"] for r in kept})
    types = sorted({r["target_type"] for r in kept})
    classes = sorted({maneuver_class(r) for r in kept})
    data = {
        "about": ("Shots for tests/test_fire_control_calibration.py, written by "
                  "scripts/fire_control_calibration.py fixture. class is the target's maneuver "
                  "class at fire time; n is how many ships had fired at it in the last 30 s "
                  "(harness: the shooter count); dodge (its evasion throttle, null = does not "
                  "dodge), jink (missed jink burn, m/s^2 across the line of fire) "
                  "and pending / pending_in (unlit HEADING burn, m/s^2, s) are the forecast's "
                  "TargetMotion inputs; p_conv is the engine's conversion (dispersion-roll) "
                  "probability; hit is the engine outcome."),
        "columns": FIXTURE_COLUMNS,
        "sources": sources,
        "target_types": types,
        "classes": classes,
        "rows": [[sources.index(r["source"]), round(r["range_km"], 1), round(r["tof_s"], 2),
                  types.index(r["target_type"]), classes.index(maneuver_class(r)),
                  None if r.get("dodge") is None else round(r["dodge"], 2), int(r.get("n") or 1),
                  round(r.get("jink") or 0.0, 2), round(r.get("pending") or 0.0, 2),
                  round(r.get("pending_in") or 0.0, 1),
                  round(r["p_conv"], 4), int(r["outcome"] == "hit")] for r in kept],
    }
    rows_out = data.pop("rows")
    head = json.dumps(data, indent=1)[:-2]          # leave the object open
    with open(out, "w") as f:
        f.write(head + ',\n "rows": [\n')
        f.write(",\n".join(json.dumps(r, separators=(",", ":")) for r in rows_out))
        f.write("\n ]\n}\n")
    print(f"wrote {len(kept)} shots ({sum(1 for r in kept if r.get('dodge') is not None)} "
          f"at dodging targets) to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("recordings", help="per-shot rows from battle recordings")
    a.add_argument("paths", nargs="+")
    a.add_argument("--out", required=True)
    a.add_argument("--source", help="source name (default: the file name; one path only)")
    h = sub.add_parser("harness", help="controlled engagements on the live engine")
    h.add_argument("--runs", type=int, default=80, help="engagements per target class and shooter count")
    h.add_argument("--classes", default=",".join(HARNESS_CLASSES))
    h.add_argument("--shooters", default="1", help="comma list of shooter counts")
    h.add_argument("--workers", type=int, default=1)
    h.add_argument("--out", required=True)
    t = sub.add_parser("table", help="calibration table old vs current model")
    t.add_argument("paths", nargs="+")
    t.add_argument("--by", choices=("range", "fleet", "source"), default="range")
    fx = sub.add_parser("fixture", help="write the compact test fixture")
    fx.add_argument("paths", nargs="+")
    fx.add_argument("--out", required=True)
    fx.add_argument("--free-keep", type=int, default=4)
    fx.add_argument("--harness-keep", type=int, default=20)
    fx.add_argument("--pre-lead-fix", default="",
                    help="comma list of sources recorded before the HEADING lead fix")
    args = ap.parse_args()

    if args.cmd == "recordings":
        n = 0
        hulls = HullTable()
        with open(args.out, "w") as f:
            for p in args.paths:
                rows = rows_from_recording(p, hulls)
                n += len(rows)
                print(f"{len(rows):5d} shots  {p}")
                for r in rows:
                    if args.source:
                        r["source"] = args.source
                    f.write(json.dumps(r) + "\n")
        print(f"{n} shots -> {args.out}")
    elif args.cmd == "harness":
        jobs = harness_jobs(args.runs, args.classes.split(","),
                            [int(x) for x in args.shooters.split(",")])
        with open(args.out, "w") as f:
            if args.workers > 1:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(args.workers) as pool:
                    results = pool.map(_harness_job, jobs, chunksize=2)
                    for rows in results:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
            else:
                for job in jobs:
                    for r in _harness_job(job):
                        f.write(json.dumps(r) + "\n")
        print(f"{len(jobs)} engagements -> {args.out}")
    elif args.cmd == "table":
        print_table(calibration_table(read_rows(args.paths), HullTable(), args.by))
    elif args.cmd == "fixture":
        write_fixture(read_rows(args.paths), args.out, args.free_keep, args.harness_keep,
                      [s for s in args.pre_lead_fix.split(",") if s])


if __name__ == "__main__":
    main()
