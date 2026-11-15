#!/usr/bin/env python3
"""
Round-robin draft-balance tournament: is any fleet concept dominant at equal points?

Eight fleet archetypes are drafted to the SAME point budget under a price
table and fight each other with the rule-based (tactical heuristic) captains,
no admirals, no LLM. Every pairing is fought in both side assignments
(alpha/beta swapped) per seed, so side bias cancels.

    torpedo     auto_draft "torpedo"  (torpedo cruisers, corvettes, a frigate)
    gunline     auto_draft "gunline"  (destroyers and cruisers, frigates)
    combined    auto_draft "combined" (battleship / destroyer / corvette)
    flagship    auto_draft "flagship" (dreadnoughts with frigates and a corvette)
    battleline  battleships, then destroyers
    escort      two battleships behind a frigate PD screen
    swarm       corvettes to the hull cap
    screen      one torpedo cruiser behind a frigate screen

The price table decides WHICH roster an archetype fields and what a hull is
worth in a time-limit verdict; nothing else in a battle reads a price. So a
battle is determined by (engine source, rosters, the prices of their hulls,
seed, battle settings) and is cached under that key in --out: re-scoring a
candidate price table costs only the battles whose rosters or hull prices it
changed.

    # the current prices at 150 pts, one seed, both sides (42 battles)
    uv run python scripts/draft_tournament.py --budget 150 --seeds 1 --out t.jsonl

    # a candidate price table (any pricing constant of src/llm/fleet_draft.py)
    uv run python scripts/draft_tournament.py --set TORPEDO_POINTS_PER_GJ=0.05 \\
        --budget 150 --seeds 1 --out t.jsonl

    # only the pairings that involve these archetypes, more seeds
    uv run python scripts/draft_tournament.py --rows torpedo,swarm --seeds 3 --out t.jsonl

    # rosters only / report from the cache only
    uv run python scripts/draft_tournament.py --dry-run
    uv run python scripts/draft_tournament.py --out t.jsonl --summary-only

Reuses the battle plumbing of scripts/calibrate_torpedo_cost.py (engine
loading, fork pool, JSONL cache). Threads are pinned to 1 per battle.
"""

from __future__ import annotations

import os
import sys

_PINNED_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
if __name__ == "__main__" and any(
        os.environ.get(k) != v for k, v in _PINNED_ENV.items()):
    os.environ.update(_PINNED_ENV)
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import hashlib
import json
import multiprocessing as mp
import signal
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_torpedo_cost as calib  # noqa: E402  (shared battle plumbing)

ROOT = Path(__file__).resolve().parent.parent

# auto: an auto_draft doctrine by name. Otherwise calibrate_torpedo_cost's
# recipe (core / cycle / top_up) plus `upgrade`: once the hull cap binds, swap
# the cheapest hull for the dearest of these the leftover points allow
# (auto_draft's own rule), so every archetype spends its budget.
ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "torpedo": {"auto": "torpedo"},
    "gunline": {"auto": "gunline"},
    "combined": {"auto": "combined"},
    "flagship": {"auto": "flagship"},
    "battleline": {"core": [("battleship", "max")], "cycle": ["destroyer"],
                   "top_up": ["frigate"], "upgrade": ["battleship", "destroyer"]},
    "escort": {"core": [("battleship", 2)], "cycle": ["frigate"], "top_up": [],
               "upgrade": ["battleship", "cruiser", "destroyer"]},
    "swarm": {"core": [("corvette", "max")], "cycle": [], "top_up": ["frigate"],
              "upgrade": ["cruiser_torpedo"]},
    # A torpedo cruiser behind a frigate screen. Added in round 2: a fleet of
    # torpedo hulls ALONE surrenders once its tubes are empty (it has no gun
    # left), so "torpedo" and "swarm" understate torpedoes whenever their
    # recipe happens to buy no gun hull. This one always has guns.
    "screen": {"core": [("cruiser_torpedo", 1)], "cycle": ["frigate"], "top_up": [],
               "upgrade": ["destroyer"]},
}


def price_table(overrides: Dict[str, float]) -> Dict[str, int]:
    """Hull costs with fleet_draft pricing constants overridden (restored after)."""
    fd = calib.ENGINE["fleet_draft"]
    saved = {k: getattr(fd, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(fd, k, v)
        return fd.build_ship_point_costs(calib.ENGINE["fleet_data"])
    finally:
        for k, v in saved.items():
            setattr(fd, k, v)


def draft_roster(name: str, budget: int, costs: Dict[str, int], max_ships: int) -> List[str]:
    recipe = ARCHETYPES[name]
    fd = calib.ENGINE["fleet_draft"]
    if "roster" in recipe:          # --fleet: an explicit roster, any price
        return list(recipe["roster"])
    if "auto" in recipe:
        live = fd.SHIP_POINT_COSTS
        saved = dict(live)
        try:
            live.clear()
            live.update(costs)
            draft = fd.auto_draft("alpha", budget=budget, max_ships=max_ships,
                                  doctrine=recipe["auto"])
        finally:
            live.clear()
            live.update(saved)
        return [s.ship_type for s in draft.ships]
    saved_recipes = calib.ARCHETYPES
    try:
        calib.ARCHETYPES = {name: recipe}
        picked = calib.draft_archetype(name, budget, costs, max_ships)
    finally:
        calib.ARCHETYPES = saved_recipes
    pool = recipe.get("upgrade") or []
    spent = sum(costs[h] for h in picked)
    while len(picked) >= max_ships and pool:
        i = min(range(len(picked)), key=lambda k: (costs[picked[k]], k))
        cheapest = costs[picked[i]]
        ups = [h for h in pool if cheapest < costs[h] <= budget - spent + cheapest]
        if not ups:
            break
        best = max(ups, key=lambda h: (costs[h], -pool.index(h)))
        spent += costs[best] - cheapest
        picked[i] = best
    return picked


def source_id(root: Path) -> str:
    """Identity of the battle engine: content hash of src/ and the ship data,
    without fleet_draft.py (prices are never read by a battle). Unlike a git
    HEAD it survives commits that do not touch the engine."""
    h = hashlib.sha1()
    files = sorted(p for p in (root / "src").rglob("*.py")
                   if p.name != "fleet_draft.py")
    files.append(root / "data" / "fleet_ships.json")
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# One battle
# ---------------------------------------------------------------------------

def run_battle(task: Dict[str, Any]) -> Dict[str, Any]:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    signal.signal(signal.SIGALRM, calib._timeout)
    signal.alarm(int(task["battle_timeout"]))
    started = time.time()
    cpu0 = time.process_time()
    out = dict(task)
    try:
        E = calib.ENGINE
        fd = E["fleet_draft"]
        drafts = {}
        for faction in ("alpha", "beta"):
            roster = task[f"{faction}_roster"]
            draft = fd.FleetDraft(faction=faction, budget=task["budget"], auto=True,
                                  formation_name="staggered wall")
            draft.ships = fd.name_drafted_ships(list(roster), faction)
            for ship, off in zip(draft.ships, calib.formation_offsets("staggered", roster, {})):
                ship.offset_km = tuple(off)
            drafts[faction] = draft
        dist = task["distance_km"]
        kw = calib._dataclass_kwargs
        fleet_cfg = E["BattleFleetConfig"](**kw(
            E["BattleFleetConfig"], battle_name=f"tourney {task['key']}",
            alpha_fleet=fd.draft_to_fleet_definition(
                drafts["alpha"], dist, captain_model=E["HEURISTIC_MODEL"]),
            beta_fleet=fd.draft_to_fleet_definition(
                drafts["beta"], dist, captain_model=E["HEURISTIC_MODEL"]),
            time_limit_s=task["time_limit_s"], initial_distance_km=dist,
            record_battle=False, record_sim_trace=False,
            personality_selection=False, use_notebooks=False))
        battle_cfg = E["BattleConfig"](**kw(
            E["BattleConfig"], initial_distance_km=dist,
            time_limit_s=task["time_limit_s"], max_checkpoints=task["max_checkpoints"],
            verbose=False, personality_selection=False, record_battle=False,
            record_sim_trace=False, record_predictions=False, seed=task["seed"]))
        dummy = E["LLMCaptainConfig"](
            name="-", ship_name="-", model=E["HEURISTIC_MODEL"],
            personality=E["CaptainPersonality"].BALANCED)
        runner = E["LLMBattleRunner"](config=battle_cfg, alpha_config=dummy,
                                      beta_config=dummy, client=None,
                                      fleet_config=fleet_cfg)
        result = runner.run_battle(E["fleet_data"])

        ships = {**runner.alpha_ships, **runner.beta_ships}
        hull = {sid: getattr(s, "ship_type", "?") for sid, s in ships.items()}

        def side(sid: Optional[str]) -> str:
            return (sid or "").split("_", 1)[0]

        ET = E["EventType"]
        ev: Counter = Counter()
        firsts: Dict[str, float] = {}
        min_shot_km = {"alpha": None, "beta": None}
        deaths: List[Dict[str, Any]] = []
        seen_dead = set()
        launches_by_hull: Counter = Counter()

        def first(name: str, t: float) -> None:
            firsts.setdefault(name, round(t, 1))

        for e in runner.simulation.events:
            s, et, d = side(e.ship_id), e.event_type, (e.data or {})
            if et == ET.TORPEDO_LAUNCHED:
                ev[f"{s}_torp_launched"] += 1
                launches_by_hull[f"{s}:{hull.get(e.ship_id, '?')}"] += 1
                first(f"{s}_first_launch", e.timestamp)
            elif et == ET.TORPEDO_IMPACT:           # ship_id = launching ship
                ev[f"{s}_torp_impacts"] += 1
                ev[f"{s}_torp_gj"] += d.get("total_damage_gj", 0.0)
                first(f"{s}_first_torp_hit", e.timestamp)
            elif et == ET.PD_TORPEDO_DISABLED:      # ship_id = defending ship
                ev[f"{s}_pd_blinded"] += 1
            elif et == ET.PD_TORPEDO_DESTROYED:
                ev[f"{s}_pd_destroyed"] += 1
            elif et == ET.PROJECTILE_LAUNCHED:
                ev[f"{s}_shots"] += 1
                first(f"{s}_first_shot", e.timestamp)
                km = d.get("distance_km")
                if km is not None and (min_shot_km[s] is None or km < min_shot_km[s]):
                    min_shot_km[s] = round(km, 1)
            elif et == ET.PROJECTILE_IMPACT:        # ship_id = firing ship
                ev[f"{s}_gun_hits"] += 1
                ev[f"{s}_gun_gj"] += d.get("kinetic_energy_gj", 0.0)
            elif et in (ET.SHIP_DYING, ET.SHIP_DESTROYED) and e.ship_id not in seen_dead:
                seen_dead.add(e.ship_id)
                deaths.append({"t": round(e.timestamp, 1), "hull": hull.get(e.ship_id, "?"),
                               "side": s, "cause": d.get("cause"),
                               "killer": hull.get(d.get("killer_id"), None)})
        lost = {}
        for faction, fleet in (("alpha", runner.alpha_ships), ("beta", runner.beta_ships)):
            lost[faction] = sorted(
                getattr(s, "ship_type", "?") for s in fleet.values()
                if s.is_destroyed or getattr(s, "is_dying", False)
                or getattr(s, "is_surrendered", False))
        # Range between the closest pair of opposing survivors at the end
        live = {f: [s for s in fleet.values() if not s.is_destroyed
                    and not getattr(s, "is_dying", False)]
                for f, fleet in (("alpha", runner.alpha_ships), ("beta", runner.beta_ships))}
        gaps = [(a.position - b.position).magnitude / 1000.0
                for a in live["alpha"] for b in live["beta"]]
        in_flight: Counter = Counter()
        for t in getattr(runner.simulation, "torpedoes", None) or []:
            in_flight[side(getattr(t, "source_ship_id", None))] += 1
        out.update(
            torps_in_sim_at_end=dict(in_flight),
            end_range_km=round(min(gaps), 1) if gaps else None,
            rounds_left={f: sum(getattr(s, "torpedoes_remaining", 0) for s in ships_)
                         for f, ships_ in live.items()},
            winner=result.winner, reason=result.reason,
            duration_s=round(result.duration_s, 1),
            events={k: round(v, 2) for k, v in ev.items()}, firsts=firsts,
            min_shot_km=min_shot_km, deaths=deaths,
            launches_by_hull=dict(launches_by_hull),
            alpha_lost=lost["alpha"], beta_lost=lost["beta"], error=None)
    except BaseException as exc:  # recorded, never fatal to the sweep
        out.update(winner=None, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-2000:])
    finally:
        signal.alarm(0)
    out["wall_s"] = round(time.time() - started, 1)
    out["cpu_s"] = round(time.process_time() - cpu0, 1)
    out.pop("battle_timeout", None)
    return out


# ---------------------------------------------------------------------------
# Plan / report
# ---------------------------------------------------------------------------

def battle_key(src: str, alpha: Sequence[str], beta: Sequence[str], seed: int,
               args: argparse.Namespace, costs: Optional[Dict[str, int]] = None) -> str:
    # The time-limit verdict values hulls at their draft price
    # (battle_runner.ship_class_value), so the prices of the hulls IN the
    # battle are part of its identity; a reprice still reuses every battle
    # between fleets whose hulls it did not touch.
    prices = sorted((h, costs[h]) for h in set(alpha) | set(beta)) if costs else []
    raw = json.dumps([src, list(alpha), list(beta), seed, args.distance,
                      args.time_limit, args.max_checkpoints, prices])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def plan(args: argparse.Namespace, src: str, costs: Dict[str, int]):
    names = args.archetypes
    cells, tasks, fleets = [], {}, {}
    for budget in args.budget:
        rosters = {a: draft_roster(a, budget, costs, args.max_ships) for a in names}
        fleets[budget] = rosters
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if args.rows and a not in args.rows and b not in args.rows:
                    continue
                versus = getattr(args, "versus", None)
                if versus and not ((a in (args.rows or names) and b in versus)
                                   or (b in (args.rows or names) and a in versus)):
                    continue
                if sorted(rosters[a]) == sorted(rosters[b]):
                    continue  # two archetypes that draft the same fleet
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    for a_side in ("alpha", "beta"):
                        alpha, beta = ((rosters[a], rosters[b]) if a_side == "alpha"
                                       else (rosters[b], rosters[a]))
                        key = battle_key(src, alpha, beta, seed, args, costs)
                        cells.append(dict(budget=budget, a=a, b=b, seed=seed,
                                          a_side=a_side, key=key))
                        tasks.setdefault(key, dict(
                            key=key, src=src, budget=budget, seed=seed,
                            alpha_roster=list(alpha), beta_roster=list(beta),
                            distance_km=args.distance, time_limit_s=args.time_limit,
                            max_checkpoints=args.max_checkpoints,
                            battle_timeout=args.battle_timeout))
    return cells, tasks, fleets


def end_kind(r: Dict[str, Any]) -> str:
    """A: one fleet annihilated; I: one fleet combat-ineffective; T: time limit, on points."""
    reason = r.get("reason") or ""
    if "eliminated" in reason or "Mutual destruction" in reason:
        return "A"
    if "combat-ineffective" in reason:
        return "I"
    return "T"


def _pct(x: Optional[float]) -> str:
    return "  - " if x is None else f"{100 * x:3.0f}%"


def summarize(cells, results, fleets, costs, args) -> Dict[str, Any]:
    names = args.archetypes
    # Dry mass per hull: the price-independent yardstick for "how much of the
    # fleet was destroyed", so before/after price tables compare like for like.
    tons = {h: float(spec.get("performance", {}).get("max_dry_mass_tons", 1.0))
            for h, spec in calib.ENGINE["fleet_data"].get("ships", {}).items()}
    report: Dict[str, Any] = {"budgets": {}}
    for budget in args.budget:
        rosters = fleets[budget]
        print(f"\n=== budget {budget} ===")
        for a in names:
            print(f"  {a:10s} {calib.roster_str(rosters[a], costs)}")
        # score[a][b] = list of a's results vs b (1 / 0.5 / 0)
        ended: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        score: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        margin: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        diag: Dict[str, Counter] = defaultdict(Counter)
        side_wins: Counter = Counter()
        reasons: Counter = Counter()
        missing = 0
        for c in (c for c in cells if c["budget"] == budget):
            r = results.get(c["key"])
            if r is None:
                missing += 1
                continue
            for me, opp, my_side in ((c["a"], c["b"], c["a_side"]),
                                     (c["b"], c["a"], "beta" if c["a_side"] == "alpha" else "alpha")):
                other = "beta" if my_side == "alpha" else "alpha"
                s = 1.0 if r["winner"] == my_side else 0.5 if r["winner"] is None else 0.0
                score[me][opp].append(s)
                mine = sum(tons[h] for h in r[f"{my_side}_lost"]) / max(1, sum(tons[h] for h in rosters[me]))
                theirs = sum(tons[h] for h in r[f"{other}_lost"]) / max(1, sum(tons[h] for h in rosters[opp]))
                margin[me][opp].append(theirs - mine)
                ended[me][opp].append(end_kind(r))
                ev = r.get("events", {})
                d = diag[me]
                d["battles"] += 1
                d["launched"] += ev.get(f"{my_side}_torp_launched", 0)
                d["torp_hits"] += ev.get(f"{my_side}_torp_impacts", 0)
                d["torp_gj"] += ev.get(f"{my_side}_torp_gj", 0)
                blinded = ev.get(f"{other}_pd_blinded", 0)
                d["blinded"] += blinded
                d["pd_destroyed"] += ev.get(f"{other}_pd_destroyed", 0)
                # Rounds that hit although PD had blinded them (lower bound:
                # every round PD never touched is assumed to have hit).
                d["blinded_hits"] += max(0, ev.get(f"{my_side}_torp_impacts", 0)
                                         - (ev.get(f"{my_side}_torp_launched", 0) - blinded
                                            - ev.get(f"{other}_pd_destroyed", 0)))
                d["shots"] += ev.get(f"{my_side}_shots", 0)
                d["gun_hits"] += ev.get(f"{my_side}_gun_hits", 0)
                d["gun_gj"] += ev.get(f"{my_side}_gun_gj", 0)
                d["kills"] += len(r[f"{other}_lost"])
                d["losses"] += len(r[f"{my_side}_lost"])
                d["pts_killed"] += sum(costs[h] for h in r[f"{other}_lost"])
                d["pts_lost"] += sum(costs[h] for h in r[f"{my_side}_lost"])
            if r["winner"]:
                side_wins[r["winner"]] += 1
            reasons[{"A": "annihilation", "I": "combat-ineffective",
                     "T": "time limit"}[end_kind(r)]] += 1

        def cell(a: str, b: str, table) -> Optional[float]:
            v = table[a].get(b)
            return sum(v) / len(v) if v else None

        for title, table in (("WIN SCORE (W + D/2) / N, row vs column", score),
                             ("TONNAGE MARGIN: share of enemy dry mass destroyed minus share of "
                              "own lost (price-independent)", margin)):
            print(f"\n  {title}")
            print("  " + " " * 11 + " ".join(f"{b[:8]:>8s}" for b in names) + "    field   n")
            for a in names:
                vals = [x for b in names for x in table[a].get(b, [])]
                field = sum(vals) / len(vals) if vals else None
                if table is score:
                    row = " ".join(f"{_pct(cell(a, b, table)):>8s}" for b in names)
                    print(f"  {a:10s} {row}    {_pct(field)}  {len(vals):3d}")
                else:
                    row = " ".join(
                        f"{'-' if cell(a, b, table) is None else format(cell(a, b, table), '+.2f'):>8s}"
                        for b in names)
                    print(f"  {a:10s} {row}    {'-' if field is None else format(field, '+.2f')}  {len(vals):3d}")
        print("\n  W-D-L per cell (row vs column)")
        for a in names:
            print(f"  {a:10s} " + " ".join(
                f"{'-' if not score[a].get(b) else '%d-%d-%d' % (score[a][b].count(1.0), score[a][b].count(0.5), score[a][b].count(0.0)):>8s}"
                for b in names))
        print("\n  HOW IT ENDED (row vs column): A annihilation, I one side combat-ineffective, "
              "T time limit on points")
        for a in names:
            print(f"  {a:10s} " + " ".join(
                f"{'-' if not ended[a].get(b) else ''.join(f'{n}{k}' for k, n in sorted(Counter(ended[a][b]).items())):>8s}"
                for b in names))
        print("\n  per archetype, summed over its battles (blind>hit: rounds that hit although "
              "PD had blinded them, lower bound):")
        print("  archetype   torp L   hit  hit/L  PD-blinded/L  blind>hit   shots  gun hit%   kills  losses  pts killed/lost")
        for a in names:
            d = diag[a]
            if not d["battles"]:
                continue
            L = d["launched"]
            print(f"  {a:10s} {L:7d} {d['torp_hits']:5d} "
                  f"{(d['torp_hits'] / L if L else 0):6.2f} {(d['blinded'] / L if L else 0):13.2f} "
                  f"{d['blinded_hits']:10d} "
                  f"{d['shots']:7d} {(100 * d['gun_hits'] / d['shots'] if d['shots'] else 0):8.1f}% "
                  f"{d['kills']:7d} {d['losses']:7d}  {d['pts_killed']:5d}/{d['pts_lost']:<5d}")
        print(f"\n  ended: {dict(reasons)}   side check: alpha {side_wins['alpha']} / "
              f"beta {side_wins['beta']} wins" + (f"   ({missing} battles MISSING)" if missing else ""))
        report["budgets"][budget] = {
            "rosters": rosters,
            "score": {a: {b: cell(a, b, score) for b in names} for a in names},
            "margin": {a: {b: cell(a, b, margin) for b in names} for a in names},
            "n": {a: {b: len(score[a].get(b, [])) for b in names} for a in names},
            "field": {a: (lambda v: sum(v) / len(v) if v else None)(
                [x for b in names for x in score[a].get(b, [])]) for a in names},
            "diag": {a: dict(diag[a]) for a in names}, "missing": missing,
            "ended": {a: {b: dict(Counter(ended[a].get(b, []))) for b in names} for a in names},
            "reasons": dict(reasons), "side_wins": dict(side_wins)}
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="overrides", default="",
                    help="Pricing constants of fleet_draft to override, NAME=VALUE,... "
                         "(default: the checked-in prices)")
    ap.add_argument("--budget", type=calib._ints, default=[150])
    ap.add_argument("--seeds", type=int, default=1,
                    help="Seeds per pairing; each is fought twice, sides swapped")
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--archetypes", type=lambda s: s.split(","), default=list(ARCHETYPES))
    ap.add_argument("--fleet", action="append", default=[], metavar="NAME=HULL*N+HULL*N",
                    help="Add an explicit-roster archetype (not budget-checked), e.g. "
                         "--fleet six=corvette*6+frigate*2; repeatable. Name it in "
                         "--archetypes/--rows like any other")
    ap.add_argument("--rows", type=lambda s: s.split(","), default=None,
                    help="Only fight pairings that involve one of these archetypes")
    ap.add_argument("--versus", type=lambda s: s.split(","), default=None,
                    help="With --rows: only fight a --rows archetype against one of these "
                         "(skips the pairings among the rows themselves)")
    ap.add_argument("--max-ships", type=int, default=8)
    ap.add_argument("--distance", type=float, default=500.0)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--max-checkpoints", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--battle-timeout", type=int, default=3 * 3600)
    ap.add_argument("--out", type=Path, default=None, help="JSONL battle cache")
    ap.add_argument("--summary-json", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    calib.load_engine(ROOT)
    fd = calib.ENGINE["fleet_draft"]
    for spec in args.fleet:
        name, _, body = spec.partition("=")
        roster: List[str] = []
        for part in filter(None, body.split("+")):
            hull, _, n = part.partition("*")
            if hull not in fd.SHIP_POINT_COSTS:
                ap.error(f"--fleet {name}: unknown hull {hull!r}")
            roster += [hull] * int(n or 1)
        ARCHETYPES[name] = {"roster": roster}
    overrides: Dict[str, float] = {}
    for item in filter(None, (x.strip() for x in args.overrides.split(","))):
        name, _, value = item.partition("=")
        if not hasattr(fd, name):
            ap.error(f"fleet_draft has no pricing constant {name!r}")
        overrides[name] = float(value)
    for a in args.archetypes + (args.rows or []):
        if a not in ARCHETYPES:
            ap.error(f"unknown archetype {a!r} (one of {', '.join(ARCHETYPES)})")
    costs = price_table(overrides)
    # The battles must be scored at the candidate prices too (the workers
    # fork from this process): make the overrides the live table.
    for name, value in overrides.items():
        setattr(fd, name, value)
    fd.SHIP_POINT_COSTS.clear()
    fd.SHIP_POINT_COSTS.update(costs)
    src = source_id(ROOT)
    print(f"engine source {src}   prices {overrides or '(checked in)'}")
    print("  " + ", ".join(f"{h} {c}" for h, c in sorted(costs.items(), key=lambda kv: kv[1])))

    cells, tasks, fleets = plan(args, src, costs)
    results = calib.load_results(args.out)
    todo = [t for key, t in tasks.items() if key not in results]
    print(f"cells: {len(cells)} battles, {len(tasks)} unique, "
          f"{len(tasks) - len(todo)} cached, {len(todo)} to fight")
    if args.dry_run:
        for budget, rosters in fleets.items():
            print(f"\nbudget {budget}:")
            for a, r in rosters.items():
                print(f"  {a:10s} {calib.roster_str(r, costs)}")
        return 0

    started = time.time()
    if todo and not args.summary_only:
        rows = args.rows or []
        cell_of = {c["key"]: c for c in cells}
        # Requested rows first, then the heaviest battles
        todo.sort(key=lambda t: (not (cell_of[t["key"]]["a"] in rows or cell_of[t["key"]]["b"] in rows),
                                 -(len(t["alpha_roster"]) + len(t["beta_roster"]))))
        fh = open(args.out, "a") if args.out else None
        errors = 0
        with mp.get_context("fork").Pool(processes=min(args.workers, len(todo)),
                                         maxtasksperchild=1) as pool:
            for i, rec in enumerate(pool.imap_unordered(run_battle, todo), 1):
                if rec.get("error"):
                    errors += 1
                else:
                    results[rec["key"]] = rec
                if fh:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                c = cell_of[rec["key"]]
                print(f"[{i}/{len(todo)} {time.time() - started:5.0f}s] {c['a']}({c['a_side']}) v "
                      f"{c['b']} seed {rec['seed']} winner={rec.get('winner')} "
                      f"cpu={rec.get('cpu_s')}s wall={rec['wall_s']}s"
                      + (f" ERROR {rec['error']}" if rec.get("error") else ""), flush=True)
        if fh:
            fh.close()
        print(f"fought {len(todo)} battles ({errors} errors) in {time.time() - started:.0f}s wall")

    report = summarize(cells, results, fleets, costs, args)
    report.update(source=src, overrides=overrides, costs=costs, seeds=args.seeds,
                  seed_start=args.seed_start, time_limit_s=args.time_limit)
    if args.summary_json:
        args.summary_json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
