#!/usr/bin/env python3
"""
Fire-control calibration: is the per-shot P(hit) commanders see what the
engine actually delivers?

P(hit) = conversion x track (src/firecontrol.py). This script measures it
against what the engine does, in two ways, and writes the test fixture:

  recordings   Per-shot dataset from battle recordings (recording_version 3:
               shot_fired events with projectile_id, and their hit / miss
               events). Features: range, time of flight, target hull, whether
               the target's evasion was armed (EVASIVE or the evade overlay)
               and at what throttle, the conversion probability (recorded as
               conversion_probability; older recordings recorded only
               hit_probability, which WAS the conversion number), outcome
               (hit / roll = lost the dispersion roll / gate = passed outside
               the 500 m gate).
  harness      Controlled engagements on the live engine: a PADLOCKed
               battleship firing FIRE_IMMEDIATE at one target that coasts,
               PADLOCKs, burns on a HEADING, flies EVASIVE, or PADLOCK /
               HOLD_RANGE with evade='dodge'; frigate..dreadnought, 40-400 km.
               Seeded and deterministic; this is what calibrated the dodge.
  table        Calibration table: observed hit rate vs the recorded (old) and
               the current model's P(hit), per target class x 100 km bucket.
  fixture      Compact shot list for tests/test_fire_control_calibration.py.

Examples:
    uv run python scripts/fire_control_calibration.py recordings \\
        visualizer/public/recordings/p4_*_flash_mirror_150pt.json --out shots.jsonl
    uv run python scripts/fire_control_calibration.py harness --runs 80 --workers 6 --out harness.jsonl
    uv run python scripts/fire_control_calibration.py table shots.jsonl
    uv run python scripts/fire_control_calibration.py table harness.jsonl --by class
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
    BALLISTIC_THREAT_HORIZON_S, EVASION_MIN_JINK_S, EVASION_TURN_ALLOWANCE_S,
    CombatSimulation, Maneuver, ManeuverType, SimulationEventType,
    create_ship_from_fleet_data,
)

FLEET_PATH = ROOT / "data" / "fleet_ships.json"
HARNESS_CLASSES = ("coast", "padlock", "burn", "evasive", "padlock_dodge", "hold_dodge")
HARNESS_HULLS = ("frigate", "destroyer", "cruiser", "battleship", "dreadnought")
HARNESS_RANGES_KM = (40, 80, 120, 160, 220, 300, 400)
HARNESS_DURATION_S = 150.0


def load_fleet() -> dict:
    with open(FLEET_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model evaluation for a dataset row
# ---------------------------------------------------------------------------

class HullTable:
    """Per-class motion capability and hull radius of a fresh ship."""

    def __init__(self, fleet: Optional[dict] = None):
        self.fleet = fleet or load_fleet()
        self._sim = CombatSimulation(seed=0)
        self._cache: Dict[str, tuple] = {}

    def hull(self, ship_type: str) -> tuple:
        if ship_type not in self._cache:
            ship = create_ship_from_fleet_data("x", ship_type, "alpha", self.fleet)
            self._cache[ship_type] = (
                ship.kinematic_state.max_acceleration_ms2(),
                self._sim._estimate_turn_time(ship, math.pi / 2.0, engines_on=True),
                ship.geometry.radius_m,
            )
        return self._cache[ship_type]

    def motion(self, ship_type: str, armed: bool, throttle: Optional[float]) -> TargetMotion:
        accel, turn_90, _ = self.hull(ship_type)
        return TargetMotion(
            accel_ms2=accel, turn_90_s=turn_90, dodges=bool(armed),
            dodge_throttle=1.0 if throttle is None else max(0.0, min(1.0, throttle)),
            dodge_horizon_s=BALLISTIC_THREAT_HORIZON_S,
            dodge_min_time_s=EVASION_TURN_ALLOWANCE_S + EVASION_MIN_JINK_S,
        )

    def predict(self, row: dict) -> float:
        """The current model's P(hit) for a recorded shot: its conversion
        probability x track for that hull, armed state and flight time."""
        motion = self.motion(row["target_type"], row["armed"], row.get("throttle"))
        envelope = HIT_TOLERANCE_M + self.hull(row["target_type"])[2]
        sigma = displacement_sigma_m(row["tof_s"], motion)
        return min(0.99, row["p_conv"] * track_probability(sigma, envelope))


# ---------------------------------------------------------------------------
# recordings -> rows
# ---------------------------------------------------------------------------

def rows_from_recording(path: str) -> List[dict]:
    with open(path) as f:
        rec = json.load(f)
    name = Path(path).stem
    hulls = {}
    for side in ("alpha", "beta"):
        for ship in (rec.get(f"{side}_fleet") or {}).get("ships", []):
            hulls[ship["ship_id"]] = ship.get("ship_type")
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
    maneuvers = collections.defaultdict(list)
    for e in events:
        if e["event_type"] == "maneuver":
            maneuvers[e["ship_id"]].append((e["timestamp"], e.get("data") or {}))
    rows = []
    for e in events:
        if e["event_type"] != "shot_fired":
            continue
        d = e.get("data") or {}
        pid, target = d.get("projectile_id"), d.get("target_id")
        if pid not in outcome or hulls.get(target) is None or not d.get("eta_s"):
            continue
        p = d.get("conversion_probability", d.get("hit_probability"))
        if p is None:
            continue
        t0 = e["timestamp"]
        prior = [m for t, m in maneuvers.get(target, []) if t <= t0]
        m = prior[-1] if prior else {}
        rows.append({
            "source": name, "t": t0, "range_km": d.get("distance_km"), "tof_s": d["eta_s"],
            "target_type": hulls[target],
            "armed": m.get("maneuver_type") == "EVASIVE" or bool(m.get("evade")),
            "throttle": m.get("throttle"), "p_conv": p, "p_shown": d.get("hit_probability"),
            "outcome": outcome[pid],
        })
    return rows


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def _unit(rng: random.Random) -> Vector3D:
    while True:
        v = Vector3D(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
        if 0.1 < v.magnitude <= 1.0:
            return v.normalized()


def harness_run(seed: int, cls: str, target_type: str, range_km: float, fleet: dict) -> List[dict]:
    """One controlled engagement; returns one row per resolved round."""
    rng = random.Random(seed)
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=seed)
    shooter = create_ship_from_fleet_data("shooter", "battleship", "alpha", fleet,
                                          position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    target = create_ship_from_fleet_data(
        "target", target_type, "beta", fleet, position=Vector3D(range_km * 1000.0, 0, 0),
        velocity=_unit(rng) * rng.uniform(0, 1500),
        forward=_unit(rng) if rng.random() < 0.5 else Vector3D(-1, 0, 0))
    sim.add_ship(shooter)
    sim.add_ship(target)
    shooter.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0, duration=1e9, target_id="target")
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
    shooter.weapons_orders = {
        slot: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, slot, "target", min_hit_probability=0.0)
        for slot in shooter.weapons if not slot.startswith("pd_")}
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
                                  "p_shown": d.get("hit_probability")}
                elif pid in shots and pid not in outcome:
                    if ev.event_type == SimulationEventType.PROJECTILE_IMPACT:
                        outcome[pid] = "hit"
                    elif ev.event_type == SimulationEventType.PROJECTILE_MISS:
                        outcome[pid] = "roll" if d.get("detection") == "dispersion_roll" else "gate"
    armed = cls in ("evasive", "padlock_dodge", "hold_dodge")
    return [dict(shot, source=f"harness:{cls}", target_type=target_type, armed=armed,
                 throttle=m.throttle, outcome=outcome[pid], seed=seed)
            for pid, shot in shots.items() if pid in outcome and shot["tof_s"]]


def _harness_job(args) -> List[dict]:
    seed, cls, target_type, range_km = args
    return harness_run(seed, cls, target_type, range_km, load_fleet())


def harness_jobs(runs: int, classes: Iterable[str]) -> List[tuple]:
    rng = random.Random(12345)
    jobs = []
    for i in range(runs):
        target_type = HARNESS_HULLS[i % len(HARNESS_HULLS)]
        range_km = rng.choice(HARNESS_RANGES_KM)
        for cls in classes:
            jobs.append((1000 + i, cls, target_type, range_km))
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
    if by == "class":
        return (row["source"].split(":", 1)[-1], rb)
    if by == "source":
        return (row["source"], "armed" if row["armed"] else "free")
    return ("armed" if row["armed"] else "free", rb)


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
    print(f"{'bucket':34s} {'n':>5s} {'hits':>5s} {'obs':>6s} {'gate':>5s} "
          f"{'old P':>6s} {'old/obs':>7s} {'new P':>6s} {'new/obs':>7s}")
    for t in table:
        b = t["bucket"]
        label = (f"{b[0]} {b[1] * 100}-{b[1] * 100 + 99} km" if isinstance(b[1], int) else f"{b[0]} {b[1]}")
        print(f"{label:34s} {t['n']:5d} {t['hits']:5d} {t['observed']:6.3f} {t['gate_pass']:5.2f} "
              f"{t['old']:6.3f} {_ratio(t['old'], t['observed']):6.2f}x {t['new']:6.3f} "
              f"{_ratio(t['new'], t['observed']):6.2f}x")


FIXTURE_COLUMNS = ["source", "range_km", "tof_s", "target_type", "armed", "throttle", "p_conv", "hit"]


def write_fixture(rows: List[dict], out: str, free_keep: int, harness_keep: int) -> None:
    """Every battle shot at an armed target and every one 300 km or more out
    (rare, and where calibration matters most); an evenly spaced sample of
    the rest per source and 100 km bucket (closer battle shots at free
    targets: 1 in free_keep; harness rounds at armed targets: 1 in
    harness_keep). Harness rounds at free targets are left out: their orders
    never change, which no battle reproduces."""
    kept = []
    counters = collections.Counter()
    for r in rows:
        harness = r["source"].startswith("harness:")
        if harness and not r["armed"]:
            continue
        key = (harness, r["armed"], min(int((r["range_km"] or 0.0) // 100), 5), r["source"].split(":")[0])
        counters[key] += 1
        far = (r["range_km"] or 0.0) >= 300.0
        keep_every = harness_keep if harness else (1 if r["armed"] or far else free_keep)
        if (counters[key] - 1) % keep_every == 0:
            kept.append(r)
    sources = sorted({r["source"] for r in kept})
    types = sorted({r["target_type"] for r in kept})
    data = {
        "about": ("Shots for tests/test_fire_control_calibration.py, written by "
                  "scripts/fire_control_calibration.py fixture. p_conv is the engine's "
                  "conversion (dispersion-roll) probability; hit is the engine outcome."),
        "columns": FIXTURE_COLUMNS,
        "sources": sources,
        "target_types": types,
        "rows": [[sources.index(r["source"]), round(r["range_km"], 1), round(r["tof_s"], 2),
                  types.index(r["target_type"]), int(bool(r["armed"])),
                  None if r.get("throttle") is None else round(r["throttle"], 2),
                  round(r["p_conv"], 4), int(r["outcome"] == "hit")] for r in kept],
    }
    rows_out = data.pop("rows")
    head = json.dumps(data, indent=1)[:-2]          # leave the object open
    with open(out, "w") as f:
        f.write(head + ',\n "rows": [\n')
        f.write(",\n".join(json.dumps(r, separators=(",", ":")) for r in rows_out))
        f.write("\n ]\n}\n")
    print(f"wrote {len(kept)} shots ({sum(1 for r in kept if r['armed'])} at armed targets) to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("recordings", help="per-shot rows from battle recordings")
    a.add_argument("paths", nargs="+")
    a.add_argument("--out", required=True)
    h = sub.add_parser("harness", help="controlled engagements on the live engine")
    h.add_argument("--runs", type=int, default=80, help="engagements per target class")
    h.add_argument("--classes", default=",".join(HARNESS_CLASSES))
    h.add_argument("--workers", type=int, default=1)
    h.add_argument("--out", required=True)
    t = sub.add_parser("table", help="calibration table old vs current model")
    t.add_argument("paths", nargs="+")
    t.add_argument("--by", choices=("range", "class", "source"), default="range")
    fx = sub.add_parser("fixture", help="write the compact test fixture")
    fx.add_argument("paths", nargs="+")
    fx.add_argument("--out", required=True)
    fx.add_argument("--free-keep", type=int, default=4)
    fx.add_argument("--harness-keep", type=int, default=20)
    args = ap.parse_args()

    if args.cmd == "recordings":
        n = 0
        with open(args.out, "w") as f:
            for p in args.paths:
                rows = rows_from_recording(p)
                n += len(rows)
                print(f"{len(rows):5d} shots  {p}")
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"{n} shots -> {args.out}")
    elif args.cmd == "harness":
        jobs = harness_jobs(args.runs, args.classes.split(","))
        with open(args.out, "w") as f:
            if args.workers > 1:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(args.workers) as pool:
                    results = pool.map(_harness_job, jobs, chunksize=4)
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
        write_fixture(read_rows(args.paths), args.out, args.free_keep, args.harness_keep)


if __name__ == "__main__":
    main()
