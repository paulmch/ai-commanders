#!/usr/bin/env python3
"""
Calibrate the draft price of torpedoes (TORPEDO_POINTS_PER_GJ).

Runs paired, seeded, zero-LLM battles (heuristic captains, no admirals)
between fixed archetype fleets, each drafted to the SAME point budget at a
given TORPEDO_POINTS_PER_GJ, and reports how the torpedo fleet fares:

    torpedo   cruiser_torpedo (as many as fit) + corvettes, frigate top-up
    gunline   cruisers (as many as fit) + destroyers, frigate top-up
    dread     one dreadnought + cruisers, frigate top-up
    balanced  battleship / destroyer / corvette cycle (auto-draft's
              "combined arms"), frigate top-up

Every (opponent, seed) cell is fought twice with the fleets swapped between
alpha and beta, so side and formation bias cancel. The formation shape is
the same for both fleets in a battle and cycles with the seed (staggered
wall / line abreast / screen) unless --formation pins one.

Only the price moves with the dial - no battle code reads it - so a battle
is fully determined by (engine, rosters, formation, seed). Results are
cached in the JSONL (--out) under that key: re-runs, refinements between
two dial values that draft the same fleets, and the gun-vs-gun-invariant
opponents are never fought twice.

Examples:
    # Sweep the dial on this checkout at 150 pts, 12 paired seeds per cell
    uv run python scripts/calibrate_torpedo_cost.py \\
        --k 0.032,0.036,0.040,0.044,0.048 --budget 150 --seeds 12 \\
        --out calib.jsonl

    # Same harness against another checkout's engine (old-engine baseline)
    uv run python scripts/calibrate_torpedo_cost.py \\
        --engine-root ../baseline-main --k 0.032 --seeds 16 --out base.jsonl

    # Just print the drafted fleets / re-summarise a finished run
    uv run python scripts/calibrate_torpedo_cost.py --k 0.04 --dry-run
    uv run python scripts/calibrate_torpedo_cost.py --k 0.04 --out calib.jsonl --summary-only

The script pins OMP/OPENBLAS/MKL threads to 1 (one battle per core) and
PYTHONHASHSEED=0 (reproducible set ordering) by re-executing itself.
"""

from __future__ import annotations

import os
import sys

# One battle per core, reproducible across processes. Must happen before
# numpy is imported, hence a re-exec rather than a late os.environ tweak.
_PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
if __name__ == "__main__" and any(
        os.environ.get(k) != v for k, v in _PINNED_ENV.items()):
    os.environ.update(_PINNED_ENV)
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import random
import signal
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_ENGINE_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------
# core:   (hull, count) bought first, in order; count "max" = as many as fit
# cycle:  then cycle these hulls greedily (auto_draft's fill rule), skipping
#         any that no longer fit, until none fits
# top_up: finally spend leftovers on these, in order, until none fits
# Everything respects the draft's max-ships cap.
ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "torpedo": {
        "core": [("cruiser_torpedo", "max")],
        "cycle": ["corvette"],
        "top_up": ["frigate"],
    },
    "gunline": {
        "core": [("cruiser", "max")],
        "cycle": ["destroyer"],
        "top_up": ["frigate"],
    },
    "dread": {
        "core": [("dreadnought", 1)],
        "cycle": ["cruiser"],
        "top_up": ["frigate"],
    },
    "balanced": {
        "core": [],
        "cycle": ["battleship", "destroyer", "corvette"],
        "top_up": ["frigate"],
    },
}
TORPEDO_ARCHETYPE = "torpedo"
OPPONENTS = ("gunline", "dread", "balanced")
FORMATIONS = ("staggered", "line", "screen")


def draft_archetype(name: str, budget: int, costs: Dict[str, int],
                    max_ships: int) -> List[str]:
    """Deterministic roster for an archetype at a budget and price table."""
    recipe = ARCHETYPES[name]
    picked: List[str] = []
    left = budget

    def fits(hull: str) -> bool:
        return len(picked) < max_ships and costs[hull] <= left

    for hull, count in recipe["core"]:
        n = 0
        while (count == "max" or n < count) and fits(hull):
            picked.append(hull)
            left -= costs[hull]
            n += 1
    cycle = recipe["cycle"]
    idx = 0
    while cycle and any(fits(h) for h in cycle):
        hull = cycle[idx % len(cycle)]
        idx += 1
        if fits(hull):
            picked.append(hull)
            left -= costs[hull]
    for hull in recipe["top_up"]:
        while fits(hull):
            picked.append(hull)
            left -= costs[hull]
    return picked


def formation_offsets(shape: str, roster: Sequence[str],
                      costs: Dict[str, int]) -> List[Tuple[float, float, float]]:
    """Admiral-local offsets (+x toward the enemy), km."""
    n = len(roster)
    if shape == "staggered":
        # auto_draft's "staggered wall"
        return [(-10.0 if i % 2 else 0.0, (i - (n - 1) / 2) * 12.0,
                 6.0 if i % 3 == 2 else 0.0) for i in range(n)]
    if shape == "line":
        # fleet_draft.default_formation_offsets: line abreast, 8 km spacing
        return [(0.0, (i - (n - 1) / 2) * 8.0, 0.0) for i in range(n)]
    if shape == "screen":
        # Escorts 10 km forward, capitals and torpedo hulls 15 km back
        def heavy(h: str) -> bool:
            return h == "cruiser_torpedo" or costs.get(h, 0) >= 40
        return [(-15.0 if heavy(h) else 10.0, (i - (n - 1) / 2) * 14.0, 0.0)
                for i, h in enumerate(roster)]
    raise ValueError(f"unknown formation {shape!r}")


# ---------------------------------------------------------------------------
# Engine loading (so one harness can drive any checkout's engine)
# ---------------------------------------------------------------------------

ENGINE: Dict[str, Any] = {}


def load_engine(root: Path) -> None:
    sys.path.insert(0, str(root))
    from src.llm import fleet_draft
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    from src.llm.fleet_config import BattleFleetConfig
    from src.llm.heuristic_captain import HEURISTIC_MODEL
    from src.llm.prompts import CaptainPersonality
    from src.simulation import SimulationEventType

    loaded = Path(fleet_draft.__file__).resolve()
    if root.resolve() not in loaded.parents:
        raise RuntimeError(f"engine import resolved to {loaded}, not under {root}")
    ENGINE.update(
        root=root, fleet_draft=fleet_draft, BattleConfig=BattleConfig,
        LLMBattleRunner=LLMBattleRunner, fleet_data=load_fleet_data(),
        LLMCaptainConfig=LLMCaptainConfig, BattleFleetConfig=BattleFleetConfig,
        HEURISTIC_MODEL=HEURISTIC_MODEL, CaptainPersonality=CaptainPersonality,
        EventType=SimulationEventType,
    )


def price_table(k: float) -> Dict[str, int]:
    """Hull costs at dial value k (restores the module constant afterwards)."""
    fd = ENGINE["fleet_draft"]
    saved = fd.TORPEDO_POINTS_PER_GJ
    try:
        fd.TORPEDO_POINTS_PER_GJ = k
        return fd.build_ship_point_costs(ENGINE["fleet_data"])
    finally:
        fd.TORPEDO_POINTS_PER_GJ = saved


def engine_id(root: Path) -> str:
    """
    Identity of the battle engine: checkout path + HEAD + hash of any
    uncommitted change under src/ and data/. fleet_draft.py is excluded -
    it only prices hulls, nothing in a battle reads the price - so applying
    a new dial value does not invalidate the cache.
    """
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True,
                              text=True, check=False).stdout
    head = git("rev-parse", "HEAD").strip() or "nogit"
    diff = git("diff", "HEAD", "--", "src", "data", ":(exclude)src/llm/fleet_draft.py")
    dirty = hashlib.sha1(diff.encode()).hexdigest()[:10] if diff else "clean"
    return f"{root.resolve()}@{head[:12]}+{dirty}"


# ---------------------------------------------------------------------------
# One battle (worker)
# ---------------------------------------------------------------------------

def _dataclass_kwargs(cls: Any, **kw: Any) -> Dict[str, Any]:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in kw.items() if k in names}


def _timeout(signum: int, frame: Any) -> None:
    raise TimeoutError("battle exceeded --battle-timeout")


def run_battle(task: Dict[str, Any]) -> Dict[str, Any]:
    """Fight one battle; returns the task dict plus its outcome."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(int(task["battle_timeout"]))
    started = time.time()
    out = dict(task)
    try:
        E = ENGINE
        fd = E["fleet_draft"]
        drafts = {}
        for faction in ("alpha", "beta"):
            roster = task[f"{faction}_roster"]
            draft = fd.FleetDraft(faction=faction, budget=task["budget"], auto=True,
                                  formation_name=task["formation"])
            draft.ships = fd.name_drafted_ships(list(roster), faction)
            for ship, off in zip(draft.ships, task[f"{faction}_offsets"]):
                ship.offset_km = tuple(off)
            drafts[faction] = draft
        dist = task["distance_km"]
        fleet_cfg = E["BattleFleetConfig"](**_dataclass_kwargs(
            E["BattleFleetConfig"],
            battle_name=f"calib {task['key']}",
            alpha_fleet=fd.draft_to_fleet_definition(
                drafts["alpha"], dist, captain_model=E["HEURISTIC_MODEL"]),
            beta_fleet=fd.draft_to_fleet_definition(
                drafts["beta"], dist, captain_model=E["HEURISTIC_MODEL"]),
            time_limit_s=task["time_limit_s"], initial_distance_km=dist,
            record_battle=False, record_sim_trace=False,
            personality_selection=False, use_notebooks=False))
        battle_cfg = E["BattleConfig"](**_dataclass_kwargs(
            E["BattleConfig"],
            initial_distance_km=dist, time_limit_s=task["time_limit_s"],
            max_checkpoints=task["max_checkpoints"], verbose=False,
            personality_selection=False, record_battle=False,
            record_sim_trace=False, record_predictions=False, seed=task["seed"]))
        dummy = E["LLMCaptainConfig"](
            name="-", ship_name="-", model=E["HEURISTIC_MODEL"],
            personality=E["CaptainPersonality"].BALANCED)
        runner = E["LLMBattleRunner"](config=battle_cfg, alpha_config=dummy,
                                      beta_config=dummy, client=None,
                                      fleet_config=fleet_cfg)
        result = runner.run_battle(E["fleet_data"])

        ET = E["EventType"]
        ev = Counter()
        for e in runner.simulation.events:
            side = (e.ship_id or "").split("_", 1)[0]
            if e.event_type == ET.TORPEDO_LAUNCHED:
                ev[f"{side}_torp_launched"] += 1
            elif e.event_type == ET.TORPEDO_IMPACT:
                ev[f"{side}_torp_impacts"] += 1
            elif e.event_type == ET.PD_TORPEDO_DISABLED:
                ev[f"{side}_pd_disabled"] += 1
            elif e.event_type == ET.PD_TORPEDO_DESTROYED:
                ev[f"{side}_pd_destroyed"] += 1
        lost = {}
        for faction, ships in (("alpha", runner.alpha_ships), ("beta", runner.beta_ships)):
            lost[faction] = sorted(
                getattr(s, "ship_type", "?") for s in ships.values()
                if s.is_destroyed or getattr(s, "is_dying", False)
                or getattr(s, "is_surrendered", False))
        out.update(
            winner=result.winner, reason=result.reason,
            duration_s=round(result.duration_s, 1), events=dict(ev),
            alpha_lost=lost["alpha"], beta_lost=lost["beta"], error=None)
    except BaseException as exc:  # recorded, never fatal to the sweep
        out.update(winner=None, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-2000:])
    finally:
        signal.alarm(0)
    out["wall_s"] = round(time.time() - started, 1)
    for k in ("alpha_offsets", "beta_offsets", "battle_timeout"):
        out.pop(k, None)
    return out


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def battle_key(eng: str, alpha: Sequence[str], beta: Sequence[str],
               formation: str, seed: int, args: argparse.Namespace) -> str:
    raw = json.dumps([eng, list(alpha), list(beta), formation, seed,
                      args.distance, args.time_limit, args.max_checkpoints])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def plan(args: argparse.Namespace, eng: str):
    """Cells (budget, k, opponent, seed, torpedo side) -> battle keys + tasks."""
    cells = []
    tasks: Dict[str, Dict[str, Any]] = {}
    fleets = {}
    for budget in args.budget:
        for k in args.k:
            costs = price_table(k)
            rosters = {a: draft_archetype(a, budget, costs, args.max_ships)
                       for a in ARCHETYPES}
            fleets[(budget, k)] = (costs, rosters)
            torp = rosters[TORPEDO_ARCHETYPE]
            for opp in args.opponents:
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    shape = (FORMATIONS[seed % len(FORMATIONS)]
                             if args.formation == "cycle" else args.formation)
                    for torp_side in ("alpha", "beta"):
                        alpha, beta = ((torp, rosters[opp]) if torp_side == "alpha"
                                       else (rosters[opp], torp))
                        key = battle_key(eng, alpha, beta, shape, seed, args)
                        cells.append(dict(budget=budget, k=k, opponent=opp, seed=seed,
                                          torp_side=torp_side, key=key))
                        if key not in tasks:
                            tasks[key] = dict(
                                key=key, engine=eng, budget=budget, seed=seed,
                                formation=shape, alpha_roster=list(alpha),
                                beta_roster=list(beta),
                                alpha_offsets=formation_offsets(shape, alpha, costs),
                                beta_offsets=formation_offsets(shape, beta, costs),
                                distance_km=args.distance,
                                time_limit_s=args.time_limit,
                                max_checkpoints=args.max_checkpoints,
                                battle_timeout=args.battle_timeout)
    return cells, tasks, fleets


def load_results(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:  # truncated by an interrupted run - refight it
                continue
            if not rec.get("error"):
                done[rec["key"]] = rec
    return done


# ---------------------------------------------------------------------------
# Statistics / report
# ---------------------------------------------------------------------------

def wilson(k: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def pair_bootstrap(pairs: List[float], n_boot: int = 4000, seed: int = 7
                   ) -> Tuple[float, float]:
    """95% CI of the mean score, resampling side-swapped PAIRS (the unit)."""
    if not pairs:
        return (0.0, 1.0)
    rng = random.Random(seed)
    m = len(pairs)
    means = sorted(sum(rng.choice(pairs) for _ in range(m)) / m for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def roster_str(roster: Sequence[str], costs: Dict[str, int]) -> str:
    c = Counter(roster)
    body = ", ".join(f"{n}x {h}" if n > 1 else h for h, n in c.items())
    return f"{body} ({sum(costs[h] for h in roster)} pts)"


def summarize(cells: List[Dict[str, Any]], results: Dict[str, Dict[str, Any]],
              fleets: Dict, args: argparse.Namespace) -> Dict[str, Any]:
    by_cfg: Dict[Tuple[int, float], List[Dict[str, Any]]] = defaultdict(list)
    for c in cells:
        by_cfg[(c["budget"], c["k"])].append(c)
    report: Dict[str, Any] = {"configs": []}
    for (budget, k), cs in by_cfg.items():
        costs, rosters = fleets[(budget, k)]
        print(f"\n=== budget {budget}  TORPEDO_POINTS_PER_GJ {k:.4f}  "
              f"(cruiser_torpedo {costs['cruiser_torpedo']}, corvette {costs['corvette']}) ===")
        print(f"  torpedo : {roster_str(rosters[TORPEDO_ARCHETYPE], costs)}")
        cfg_out = {"budget": budget, "k": k, "costs": costs,
                   "rosters": rosters, "matchups": {}}
        agg = Counter()
        agg_pairs: List[float] = []
        side_wins = Counter()
        for opp in args.opponents:
            rows = [c for c in cs if c["opponent"] == opp]
            wdl = Counter()
            stats = Counter()
            per_seed: Dict[int, List[float]] = defaultdict(list)
            missing = 0
            for c in rows:
                r = results.get(c["key"])
                if r is None:
                    missing += 1
                    continue
                torp, other = c["torp_side"], ("beta" if c["torp_side"] == "alpha" else "alpha")
                if r["winner"] == torp:
                    wdl["W"] += 1
                    per_seed[c["seed"]].append(1.0)
                elif r["winner"] is None:
                    wdl["D"] += 1
                    per_seed[c["seed"]].append(0.5)
                else:
                    wdl["L"] += 1
                    per_seed[c["seed"]].append(0.0)
                if r["winner"]:
                    side_wins[r["winner"]] += 1
                ev = r.get("events", {})
                stats["launched"] += ev.get(f"{torp}_torp_launched", 0)
                stats["impacts"] += ev.get(f"{torp}_torp_impacts", 0)
                stats["pd"] += ev.get(f"{other}_pd_disabled", 0)
                stats["torp_lost"] += len(r.get(f"{torp}_lost", []))
                stats["opp_lost"] += len(r.get(f"{other}_lost", []))
                stats["torp_pts_lost"] += sum(costs[h] for h in r.get(f"{torp}_lost", []))
                stats["opp_pts_lost"] += sum(costs[h] for h in r.get(f"{other}_lost", []))
            n = sum(wdl.values())
            pairs = [sum(v) / len(v) for v in per_seed.values()]
            score = (wdl["W"] + 0.5 * wdl["D"]) / n if n else float("nan")
            lo, hi = pair_bootstrap(pairs)
            agg.update(wdl)
            agg_pairs.extend(pairs)
            imp = stats["impacts"] / stats["launched"] if stats["launched"] else 0.0
            pdr = stats["pd"] / stats["launched"] if stats["launched"] else 0.0
            print(f"  vs {opp:9s}: {roster_str(rosters[opp], costs)}")
            print(f"      {wdl['W']}W-{wdl['D']}D-{wdl['L']}L  score {score:5.1%} "
                  f"[95% CI {lo:.0%}-{hi:.0%}]  impacts/launch {imp:.3f}  "
                  f"PD-blinded/launch {pdr:.3f}  ships lost torp/opp "
                  f"{stats['torp_lost']}/{stats['opp_lost']}  pts lost "
                  f"{stats['torp_pts_lost']}/{stats['opp_pts_lost']}"
                  + (f"  ({missing} MISSING)" if missing else ""))
            cfg_out["matchups"][opp] = dict(
                W=wdl["W"], D=wdl["D"], L=wdl["L"], n=n, score=score,
                ci=[lo, hi], impacts_per_launch=imp, pd_per_launch=pdr,
                launched=stats["launched"], impacts=stats["impacts"],
                torp_ships_lost=stats["torp_lost"], opp_ships_lost=stats["opp_lost"],
                torp_pts_lost=stats["torp_pts_lost"], opp_pts_lost=stats["opp_pts_lost"],
                missing=missing)
        n = sum(agg.values())
        score = (agg["W"] + 0.5 * agg["D"]) / n if n else float("nan")
        lo, hi = pair_bootstrap(agg_pairs)
        wlo, whi = wilson(agg["W"], n)
        print(f"  AGGREGATE: {agg['W']}W-{agg['D']}D-{agg['L']}L  "
              f"win rate {agg['W'] / n if n else 0:.1%} [Wilson 95% {wlo:.0%}-{whi:.0%}]  "
              f"score (W+D/2) {score:.1%} [pair-bootstrap 95% {lo:.0%}-{hi:.0%}]  "
              f"side check: alpha {side_wins['alpha']} / beta {side_wins['beta']} wins")
        cfg_out["aggregate"] = dict(W=agg["W"], D=agg["D"], L=agg["L"], n=n,
                                    win_rate=agg["W"] / n if n else None,
                                    win_ci=[wlo, whi], score=score, score_ci=[lo, hi],
                                    alpha_wins=side_wins["alpha"],
                                    beta_wins=side_wins["beta"])
        report["configs"].append(cfg_out)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-root", type=Path, default=DEFAULT_ENGINE_ROOT,
                    help="Checkout whose src/ and data/ fight the battles "
                         "(default: this script's repo)")
    ap.add_argument("--k", type=_floats, default=None,
                    help="Comma-separated TORPEDO_POINTS_PER_GJ values "
                         "(default: the engine's current value)")
    ap.add_argument("--budget", type=_ints, default=[150],
                    help="Comma-separated point budgets (default 150)")
    ap.add_argument("--seeds", type=int, default=12,
                    help="Paired seeds per matchup cell (default 12; each is "
                         "fought twice, sides swapped)")
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--opponents", type=lambda s: s.split(","),
                    default=list(OPPONENTS),
                    help=f"Opponent archetypes (default {','.join(OPPONENTS)})")
    ap.add_argument("--formation", default="cycle",
                    choices=("cycle",) + FORMATIONS,
                    help="Formation shape for both fleets; 'cycle' rotates "
                         "through all shapes by seed (default)")
    ap.add_argument("--max-ships", type=int, default=8)
    ap.add_argument("--distance", type=float, default=500.0)
    ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--max-checkpoints", type=int, default=25)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--battle-timeout", type=int, default=3 * 3600,
                    help="Per-battle hang guard in seconds (default 3 h)")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSONL result cache (appended; reused on re-runs)")
    ap.add_argument("--summary-json", type=Path, default=None,
                    help="Write the summary tables as JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print drafted fleets and the battle count, run nothing")
    ap.add_argument("--summary-only", action="store_true",
                    help="Report from --out without fighting missing battles")
    args = ap.parse_args()

    root = args.engine_root.resolve()
    load_engine(root)
    if args.k is None:
        args.k = [ENGINE["fleet_draft"].TORPEDO_POINTS_PER_GJ]
    for a in args.opponents:
        if a not in ARCHETYPES or a == TORPEDO_ARCHETYPE:
            ap.error(f"unknown opponent archetype {a!r}")
    eng = engine_id(root)
    print(f"engine: {eng}")

    cells, tasks, fleets = plan(args, eng)
    results = load_results(args.out)
    todo = [t for key, t in tasks.items() if key not in results]
    print(f"cells: {len(cells)} battles, {len(tasks)} unique, "
          f"{len(tasks) - len(todo)} cached, {len(todo)} to fight")

    if args.dry_run:
        for (budget, k), (costs, rosters) in fleets.items():
            print(f"\nbudget {budget} k {k}:")
            for a, r in rosters.items():
                print(f"  {a:9s} {roster_str(r, costs)}")
        return 0

    started = time.time()
    if todo and not args.summary_only:
        # Heaviest first (most hulls) so the tail of the pool is short.
        todo.sort(key=lambda t: -(len(t["alpha_roster"]) + len(t["beta_roster"])))
        ctx = mp.get_context("fork")
        fh = open(args.out, "a") if args.out else None
        errors = 0
        with ctx.Pool(processes=min(args.workers, len(todo)),
                      maxtasksperchild=1) as pool:
            for i, rec in enumerate(pool.imap_unordered(run_battle, todo), 1):
                if rec.get("error"):
                    errors += 1
                else:
                    results[rec["key"]] = rec
                if fh:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                print(f"[{i}/{len(todo)} {time.time() - started:6.0f}s] "
                      f"{rec['key']} seed {rec['seed']} {rec['formation']:9s} "
                      f"winner={rec.get('winner')} wall={rec['wall_s']}s"
                      + (f" ERROR {rec['error']}" if rec.get("error") else ""),
                      flush=True)
        if fh:
            fh.close()
        print(f"fought {len(todo)} battles ({errors} errors) in "
              f"{time.time() - started:.0f}s wall")

    report = summarize(cells, results, fleets, args)
    report.update(engine=eng, seeds=args.seeds, seed_start=args.seed_start,
                  formation=args.formation, distance_km=args.distance,
                  time_limit_s=args.time_limit, max_checkpoints=args.max_checkpoints)
    if args.summary_json:
        args.summary_json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
