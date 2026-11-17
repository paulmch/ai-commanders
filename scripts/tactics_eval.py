#!/usr/bin/env python3
"""
Seeded evaluation of the free (heuristic-captain) commanders.

Runs `scripts/run_draft_battle.py --auto-draft --no-admirals` for a grid of
seeds x budgets, once per doctrine pairing, and prints a before/after table of
how the battles play: kills per side, time to first kill, closest approach
(and the closing speed there), share of gun rounds fired inside effective
range, gun hit rate, torpedoes launched / hit, ships lost to heat.

    # legacy vs legacy (before) and tactical vs tactical (after)
    uv run python scripts/tactics_eval.py --seeds 1 2 3 4 5 --budgets 120 200 \\
        --pairings legacy:legacy tactical:tactical --out /tmp/eval

    # mirror check: equal fleets (both sides draft with the same seed)
    uv run python scripts/tactics_eval.py --seeds 1 2 3 4 5 --budgets 120 200 \\
        --pairings tactical:legacy legacy:tactical --mirror --out /tmp/eval

    # analyse existing recordings only
    uv run python scripts/tactics_eval.py --analyze rec1.json.gz rec2.json

Each battle's row is cached in <out>/<tag>/metrics.json and its recording is
deleted once analysed; --keep-recordings keeps them (gzip, battle_*.json.gz).

Every battle is a separate process (BLAS threads capped), so a grid runs in
parallel; --jobs caps how many at once.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recording_io import iter_recordings, load_recording, sidecar_path  # noqa: E402

# Gun rounds fired at or inside this range count as "inside effective range":
# the doctrine's per-round P(hit) table (src/llm/prompts.py) drops below ~25%
# beyond ~150 km against anything that dodges.
EFFECTIVE_RANGE_KM = 150.0


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def analyze(path: str) -> Dict[str, Any]:
    rec = load_recording(path)
    events = rec.get("events", [])
    faction = {}
    for side in ("alpha", "beta"):
        fleet = rec.get(f"{side}_fleet") or {}
        for s in fleet.get("ships", []) if isinstance(fleet, dict) else []:
            sid = s.get("ship_id") or s.get("id")
            if sid:
                faction[sid] = side

    def side_of(sid: Optional[str]) -> Optional[str]:
        if not sid:
            return None
        return faction.get(sid) or ("alpha" if sid.startswith("alpha") else
                                    "beta" if sid.startswith("beta") else None)

    out: Dict[str, Any] = {"file": os.path.basename(path), "winner": rec.get("winner") or "draw",
                           "duration_s": rec.get("duration_s")}
    dead_at: Dict[str, float] = {}
    heat_losses = 0
    for e in events:
        et = e["event_type"]
        if et in ("ship_dying", "ship_destroyed"):
            sid = e["ship_id"]
            dead_at.setdefault(sid, e["timestamp"])
            # src/simulation.py _reactor_meltdown: the ship cooked itself.
            if et == "ship_destroyed" and e["data"].get("cause") == "reactor_overheat":
                heat_losses += 1
    out["lost_alpha"] = sum(1 for s in dead_at if side_of(s) == "alpha")
    out["lost_beta"] = sum(1 for s in dead_at if side_of(s) == "beta")
    out["kills_alpha"] = out["lost_beta"]
    out["kills_beta"] = out["lost_alpha"]
    out["first_kill_s"] = min(dead_at.values()) if dead_at else None
    out["heat_losses"] = heat_losses

    shots = [e for e in events if e["event_type"] == "shot_fired"]
    out["shots"] = len(shots)
    inside = [s for s in shots if (s["data"].get("distance_km") or 1e9) <= EFFECTIVE_RANGE_KM]
    out["shots_eff_share"] = (len(inside) / len(shots)) if shots else None
    hits = sum(1 for e in events if e["event_type"] == "hit"
               and str(e["data"].get("projectile_id", "")).startswith("proj_"))
    misses = sum(1 for e in events if e["event_type"] == "miss"
                 and str(e["data"].get("projectile_id", "")).startswith("proj_"))
    out["gun_hits"] = hits
    out["gun_misses"] = misses
    out["hit_rate"] = hits / (hits + misses) if hits + misses else None
    out["mean_fc_phit"] = (sum(s["data"].get("hit_probability") or 0.0 for s in shots) / len(shots)
                           if shots else None)
    out["torps_launched"] = sum(1 for e in events if e["event_type"] == "torpedo_launched")
    out["torps_hit"] = sum(1 for e in events if e["event_type"] == "torpedo_impact")

    # Closest approach between opposing live ships, and the closing speed
    # of that pair there (a merge is a small range at a high speed).
    best = (float("inf"), 0.0, None)
    for frame in rec.get("sim_trace") or []:
        ships = frame.get("ships", {})
        live = [(sid, s) for sid, s in ships.items()
                if not s.get("destroyed") and not s.get("dying")]
        for i, (a, sa) in enumerate(live):
            for b, sb in live[i + 1:]:
                if side_of(a) == side_of(b):
                    continue
                d = _dist(sa["pos"], sb["pos"])
                if d < best[0]:
                    rel_v = _dist(sa["vel"], sb["vel"])
                    best = (d, rel_v, frame.get("t"))
    out["closest_km"] = best[0] / 1000.0 if best[2] is not None else None
    out["closest_rel_kps"] = best[1] / 1000.0 if best[2] is not None else None
    out["closest_t"] = best[2]

    # Weapon arcs (trace 'tur'): share of gun-mount samples that bear (state 2
    # on target and clear) among those with a target (1 slewing, 2, 3 masked),
    # and gun rounds per side.
    try:
        from src.weapon_arcs import load_hull_mounts
        mounts = load_hull_mounts()
    except Exception:
        mounts = {}
    types = {}
    for side in ("alpha", "beta"):
        fleet = rec.get(f"{side}_fleet") or {}
        for s in fleet.get("ships", []) if isinstance(fleet, dict) else []:
            types[s.get("ship_id")] = s.get("ship_type")
    bear = {"alpha": [0, 0], "beta": [0, 0]}
    for frame in rec.get("sim_trace") or []:
        for sid, s in frame.get("ships", {}).items():
            tur = s.get("tur")
            hm = mounts.get(types.get(sid))
            side = side_of(sid)
            if not tur or hm is None or side not in bear:
                continue
            for m, t in zip(hm.mounts, tur):
                if m.kind in ("gun", "spinal") and len(t) > 2 and t[2] in (1, 2, 3):
                    bear[side][1] += 1
                    bear[side][0] += t[2] == 2
    tot = [bear["alpha"][0] + bear["beta"][0], bear["alpha"][1] + bear["beta"][1]]
    out["gun_bearing_share"] = tot[0] / tot[1] if tot[1] else None
    for side in ("alpha", "beta"):
        out[f"bearing_{side}"] = bear[side][0] / bear[side][1] if bear[side][1] else None
        out[f"shots_{side}"] = sum(1 for e in shots if side_of(e["ship_id"]) == side)
    return out


def run_one(seed: int, budget: int, alpha: str, beta: str, mirror: bool, out_dir: str,
            time_limit: float, extra: List[str], keep_recordings: bool = False) -> Dict[str, Any]:
    """One battle -> its analysed row. The row is cached in <tag>/metrics.json;
    the recording this call wrote (needed only for the analysis) is deleted
    afterwards unless keep_recordings - a grid of traced fleet battles is
    gigabytes. A recording already in the tag dir is analysed and kept."""
    tag = f"s{seed}_b{budget}_{alpha}-{beta}{'_m' if mirror else ''}"
    rec_dir = Path(out_dir) / tag
    rec_dir.mkdir(parents=True, exist_ok=True)
    metrics = rec_dir / "metrics.json"
    if metrics.exists():
        return json.loads(metrics.read_text())
    existing = [str(p) for p in iter_recordings(rec_dir, "battle_*")]
    ran = not existing
    if ran:
        cmd = ["uv", "run", "python", str(ROOT / "scripts" / "run_draft_battle.py"),
               "--auto-draft", "--no-admirals", "--trace", "--quiet",
               "--seed", str(seed), "--budget", str(budget), "--time-limit", str(time_limit),
               "--alpha-doctrine", alpha, "--beta-doctrine", beta,
               "--recording-dir", str(rec_dir)] + (["--mirror-draft"] if mirror else []) + extra
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        with open(rec_dir / "log.txt", "w") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        existing = [str(p) for p in iter_recordings(rec_dir, "battle_*")]
    if not existing:
        return {"tag": tag, "error": f"no recording (see {rec_dir / 'log.txt'})"}
    res = analyze(existing[-1])
    res.update(tag=tag, seed=seed, budget=budget, alpha=alpha, beta=beta)
    metrics.write_text(json.dumps(res, indent=1))
    if ran and not keep_recordings:   # never delete recordings this call did not write
        for p in existing:
            Path(p).unlink(missing_ok=True)
            sidecar_path(p).unlink(missing_ok=True)
    return res


def _fmt(v, spec="{:.0f}") -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return spec.format(v)


COLUMNS = [
    ("tag", "battle", "{}"),
    ("winner", "winner", "{}"),
    ("kills_alpha", "kA", "{}"),
    ("kills_beta", "kB", "{}"),
    ("first_kill_s", "1st kill s", "{:.0f}"),
    ("closest_km", "closest km", "{:.1f}"),
    ("closest_rel_kps", "v@closest", "{:.1f}"),
    ("shots", "rounds", "{}"),
    ("shots_eff_share", "<=150km", "{:.0%}"),
    ("hit_rate", "hit rate", "{:.1%}"),
    ("torps_launched", "torp L", "{}"),
    ("torps_hit", "torp hit", "{}"),
    ("heat_losses", "heat lost", "{}"),
]


def print_table(rows: List[Dict[str, Any]]) -> None:
    print("| " + " | ".join(c[1] for c in COLUMNS) + " |")
    print("|" + "|".join("---" for _ in COLUMNS) + "|")
    for r in rows:
        if "error" in r:
            print(f"| {r['tag']} | ERROR {r['error']} |")
            continue
        print("| " + " | ".join(_fmt(r.get(k), spec) for k, _, spec in COLUMNS) + " |")


def summarize(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)

    def mean(key):
        vals = [r[key] for r in ok if r.get(key) is not None and not
                (isinstance(r[key], float) and math.isinf(r[key]))]
        return sum(vals) / len(vals) if vals else None

    shots = sum(r["shots"] for r in ok)
    hits = sum(r["gun_hits"] for r in ok)
    misses = sum(r["gun_misses"] for r in ok)
    eff = sum(round((r["shots_eff_share"] or 0) * r["shots"]) for r in ok)
    s = {
        "label": label, "battles": n,
        "decisive": sum(1 for r in ok if r["kills_alpha"] + r["kills_beta"] > 0),
        "kills_per_battle": mean("kills_alpha") + mean("kills_beta") if n else None,
        "first_kill_s": mean("first_kill_s"),
        "no_kill": sum(1 for r in ok if r["first_kill_s"] is None),
        "closest_km": mean("closest_km"),
        "closest_rel_kps": mean("closest_rel_kps"),
        "rounds": shots,
        "eff_share": eff / shots if shots else None,
        "hit_rate": hits / (hits + misses) if hits + misses else None,
        "torps_launched": sum(r["torps_launched"] for r in ok),
        "torps_hit": sum(r["torps_hit"] for r in ok),
        "heat_losses": sum(r["heat_losses"] for r in ok),
        "alpha_wins": sum(1 for r in ok if r["winner"] == "alpha"),
        "beta_wins": sum(1 for r in ok if r["winner"] == "beta"),
        "kills_alpha": sum(r["kills_alpha"] for r in ok),
        "kills_beta": sum(r["kills_beta"] for r in ok),
    }
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--budgets", type=int, nargs="+", default=[120, 200])
    ap.add_argument("--pairings", nargs="+", default=["legacy:legacy", "tactical:tactical"],
                    help="alpha:beta doctrine pairs")
    ap.add_argument("--mirror", action="store_true", help="equal fleets (--mirror-draft)")
    ap.add_argument("--time-limit", type=float, default=420.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="data/recordings/tactics_eval")
    ap.add_argument("--json", default=None, help="write all rows + summaries here")
    ap.add_argument("--analyze", nargs="+", default=None, help="only analyse these recordings")
    ap.add_argument("--keep-recordings", action="store_true",
                    help="keep each battle's recording (default: analyse it, cache the row "
                         "in <out>/<tag>/metrics.json and delete the recording)")
    args = ap.parse_args()

    if args.analyze:
        print_table([dict(analyze(p), tag=os.path.basename(p)) for p in args.analyze])
        return 0

    jobs = [(s, b, *p.split(":"), args.mirror) for p in args.pairings
            for b in args.budgets for s in args.seeds]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda j: run_one(*j, args.out, args.time_limit, [],
                                             args.keep_recordings), jobs))
    summaries = []
    for pairing in args.pairings:
        a, b = pairing.split(":")
        sub = [r for r in rows if r.get("alpha") == a and r.get("beta") == b
               or r.get("tag", "").endswith(f"_{a}-{b}" + ("_m" if args.mirror else ""))]
        print(f"\n### {pairing}{' (mirror fleets)' if args.mirror else ''}\n")
        print_table(sub)
        summaries.append(summarize(sub, pairing))
    print("\n### summary\n")
    keys = list(summaries[0].keys())
    print("| " + " | ".join(keys) + " |")
    print("|" + "|".join("---" for _ in keys) + "|")
    for s in summaries:
        print("| " + " | ".join(
            (f"{v:.3f}" if isinstance(v, float) and v < 1 else f"{v:.1f}" if isinstance(v, float) else str(v))
            if v is not None else "-" for v in s.values()) + " |")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "summaries": summaries}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
