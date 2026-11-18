#!/usr/bin/env python3
"""
Flank experiment: does a split / pincer beat the static nose-on wall?

Runs mirror battles (equal rosters) between two heuristic doctrines
(src/llm/heuristic_flank.py: wall, pincer45/70/90, pincer70c, envelop70,
split; plus tactical / legacy), both side assignments, and measures:

- win / draw / loss, kills, time to first kill (doctrine A's view);
- where the hits land: gun hits and torpedo impacts on nose / lateral /
  tail for each side, and penetrations per hit by facing;
- the flankers (doctrine A's 'flank_*' groups): losses and hits taken while
  POSITIONING versus during the ATTACK run, rounds fired, hit rate,
  propellant used, time to reach station;
- the defender (doctrine B): share of its gun ships' samples with the nose
  within 30 deg of an attacker flanker versus of the attacker's main group.

    uv run python scripts/flank_experiment.py --pairs pincer70:wall wall:tactical \\
        --rosters gunline core_fast --budgets 150 200 --seeds 1 2 3 4 5 6 --out /tmp/flank

Each battle is cached as <out>/<tag>.json (metrics only; recordings are
written to a temp dir and deleted). --jobs processes in parallel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AUTO_ROSTERS = ("gunline", "torpedo", "combined", "flagship")
# Rosters built for the idea: a heavy core with fast hulls.
CUSTOM_ROSTERS = {
    "core_fast": {150: ["battleship", "cruiser", "destroyer", "destroyer",
                        "frigate", "frigate", "frigate", "frigate"],
                  200: ["battleship", "battleship", "cruiser", "destroyer",
                        "frigate", "frigate", "frigate", "frigate"]},
    "dread_fast": {150: ["dreadnought", "destroyer", "destroyer", "destroyer",
                         "frigate", "frigate", "frigate", "frigate"],
                   200: ["dreadnought", "battleship", "destroyer", "destroyer",
                         "frigate", "frigate", "frigate", "frigate"]},
    # Capital lines with a fast fringe (for the switch rule's thresholds).
    "battleline": {150: ["battleship", "battleship", "destroyer", "destroyer", "destroyer"],
                   200: ["battleship", "battleship", "battleship", "destroyer", "destroyer", "destroyer"]},
    "escort": {150: ["battleship", "battleship", "frigate", "frigate", "frigate",
                     "frigate", "frigate", "frigate"],
               200: ["battleship", "battleship", "battleship", "frigate", "frigate",
                     "frigate", "frigate", "frigate"]},
}
NOSE_COS = math.cos(math.radians(30.0))


def roster_types(name: str, budget: int) -> List[str]:
    if name in CUSTOM_ROSTERS:
        return list(CUSTOM_ROSTERS[name][budget])
    from src.llm.fleet_draft import auto_draft
    return [s.ship_type for s in auto_draft("alpha", budget=budget, seed=1, doctrine=name).ships]


def side_roster(job: Dict[str, Any], faction: str) -> str:
    """Roster flown by ``faction``: 'X' is a mirror; 'X/Y' gives doctrine A
    roster X and doctrine B roster Y (asymmetric families, e.g. torpedo
    against gun), whichever side A is on."""
    if "/" not in job["roster"]:
        return job["roster"]
    ra, rb = job["roster"].split("/")
    a_side = "alpha" if job["a_is_alpha"] else "beta"
    return ra if faction == a_side else rb


def side_types(job: Dict[str, Any], faction: str) -> List[str]:
    return roster_types(side_roster(job, faction), job["budget"])


def _hulls(types: List[str], faction: str) -> List[Any]:
    from src.llm.battle_runner import load_fleet_data
    from src.llm.fleet_draft import name_drafted_ships
    from src.physics import Vector3D
    from src.simulation import create_ship_from_fleet_data
    fd = load_fleet_data()
    return [create_ship_from_fleet_data(d.ship_id, d.ship_type, faction, fd,
                                        position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0),
                                        forward=Vector3D(1, 0, 0))
            for d in name_drafted_ships(types, faction)]


def spec_for(doctrine: str, types: List[str], enemy_types: Optional[List[str]] = None):
    """The FlankSpec a doctrine flies with this roster (the default doctrine
    decides it from both compositions); None = no flank plan."""
    from src.llm.heuristic_flank import choose_default_spec, flank_spec
    if doctrine == "tactical":
        return choose_default_spec(_hulls(types, "alpha"), _hulls(enemy_types or types, "beta"))
    return flank_spec(doctrine)


def groups_for(types: List[str], faction: str, doctrine: str,
               enemy_types: Optional[List[str]] = None) -> Dict[str, str]:
    """ship id -> group for a roster flown by ``doctrine`` (as the doctrine assigns it)."""
    from src.llm.fleet_draft import name_drafted_ships
    from src.llm.heuristic_flank import assign_groups
    spec = spec_for(doctrine, types, enemy_types)
    if spec is None:
        return {d.ship_id: "hold" for d in name_drafted_ships(types, faction)}
    return assign_groups(_hulls(types, faction), spec)


def run_battle(job: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out_path = Path(job["out"]) / f"{job['tag']}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_battle(job, out_path)


def _run_battle(job: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    from src.llm.fleet_config import BattleFleetConfig
    from src.llm.fleet_draft import FleetDraft, auto_draft, draft_to_fleet_definition, name_drafted_ships
    from src.llm.heuristic_captain import HEURISTIC_MODEL
    from src.llm.prompts import CaptainPersonality
    from src.recording_io import load_recording

    dist = 500.0
    drafts = {}
    for fac in ("alpha", "beta"):
        roster = side_roster(job, fac)
        if roster in CUSTOM_ROSTERS:
            d = FleetDraft(faction=fac, budget=job["budget"], auto=True)
            d.ships = name_drafted_ships(side_types(job, fac), fac)
            n = len(d.ships)
            for i, s in enumerate(d.ships):
                s.offset_km = (-10.0 if i % 2 else 0.0, (i - (n - 1) / 2) * 12.0, 0.0)
        else:
            d = auto_draft(fac, budget=job["budget"], seed=1, doctrine=roster)
        drafts[fac] = d
    tmp = tempfile.mkdtemp(prefix="flank_")
    try:
        fc = BattleFleetConfig(
            battle_name=job["tag"], time_limit_s=job["time_limit"], initial_distance_km=dist,
            alpha_fleet=draft_to_fleet_definition(drafts["alpha"], dist, captain_model=HEURISTIC_MODEL),
            beta_fleet=draft_to_fleet_definition(drafts["beta"], dist, captain_model=HEURISTIC_MODEL),
            record_battle=True, record_sim_trace=True, personality_selection=False)
        bc = BattleConfig(initial_distance_km=dist, time_limit_s=job["time_limit"], max_checkpoints=200,
                          verbose=False, personality_selection=False, record_battle=True,
                          record_sim_trace=True, record_predictions=False, recording_dir=tmp,
                          seed=job["seed"], heuristic_doctrine={"alpha": job["alpha"], "beta": job["beta"]})
        dummy = LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                                 personality=CaptainPersonality.BALANCED)
        runner = LLMBattleRunner(config=bc, alpha_config=dummy, beta_config=dummy, client=None,
                                 fleet_config=fc)
        res = runner.run_battle(load_fleet_data())
        rec = load_recording(res.recording_file)
        m = analyze(rec, job)
        m.update(reason=res.reason)
    except Exception as exc:  # recorded, never fatal to the sweep
        import traceback
        m = {"tag": job["tag"], "error": f"{type(exc).__name__}: {exc}",
             "trace": traceback.format_exc()[-1500:]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out_path.write_text(json.dumps(m))
    return m


def analyze(rec: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    a_side = "alpha" if job["a_is_alpha"] else "beta"
    b_side = "beta" if a_side == "alpha" else "alpha"
    side = lambda sid: (sid or "").split("_")[0]   # noqa: E731
    role = lambda sid: "A" if side(sid) == a_side else "B"   # noqa: E731
    events = rec.get("events", [])
    groups = {**groups_for(side_types(job, a_side), a_side, job["doc_a"], side_types(job, b_side)),
              **groups_for(side_types(job, b_side), b_side, job["doc_b"], side_types(job, a_side))}
    flankers = {sid for sid, g in groups.items() if g.startswith("flank") and role(sid) == "A"}
    a_main = {sid for sid, g in groups.items() if g == "hold" and role(sid) == "A"}

    out: Dict[str, Any] = {"tag": job["tag"], "roster": job["roster"], "budget": job["budget"],
                           "seed": job["seed"], "doc_a": job["doc_a"], "doc_b": job["doc_b"],
                           "a_side": a_side, "n_flankers": len(flankers)}
    winner = rec.get("winner")
    out["result"] = "W" if winner == a_side else ("L" if winner == b_side else "D")
    dead: Dict[str, float] = {}
    for e in events:
        if e["event_type"] in ("ship_dying", "ship_destroyed"):
            dead.setdefault(e["ship_id"], e["timestamp"])
    out["kills_a"] = sum(1 for s in dead if role(s) == "B")
    out["kills_b"] = sum(1 for s in dead if role(s) == "A")
    out["first_kill_s"] = min(dead.values()) if dead else None
    out["duration_s"] = rec.get("duration_s")

    # Attack start (doctrine A): the first decision of a flanker that is not
    # a positioning order (set_heading toward_point).
    attack_t = None
    for e in events:
        if e["event_type"] != "captain_decision" or e["ship_id"] not in flankers:
            continue
        calls = e["data"].get("tool_calls", [])
        moving = [c for c in calls if c["name"] in ("set_maneuver", "set_heading")]
        if moving and not any(c["name"] == "set_heading" and c["arguments"].get("toward_point")
                              for c in moving):
            attack_t = e["timestamp"]
            break
    out["attack_start_s"] = attack_t

    # Hit locations, by the side that TOOK the hit.
    loc = {"A": Counter(), "B": Counter()}
    pen = {"A": Counter(), "B": Counter()}
    torp = {"A": Counter(), "B": Counter()}
    shots = Counter()
    flank_hits = flank_shots = flank_misses = 0
    taken = {"position": 0, "attack": 0}
    for e in events:
        et, d = e["event_type"], e["data"]
        if et == "shot_fired":
            shots[role(e["ship_id"])] += 1
            if e["ship_id"] in flankers:
                flank_shots += 1
        elif et == "hit" and str(d.get("projectile_id", "")).startswith("proj_"):
            victim = d.get("target_id")
            r = role(victim)
            loc[r][d.get("hit_location")] += 1
            if d.get("penetrated"):
                pen[r][d.get("hit_location")] += 1
            if d.get("shooter_id") in flankers:
                flank_hits += 1
            if victim in flankers:
                taken["attack" if attack_t is not None and e["timestamp"] >= attack_t else "position"] += 1
        elif et == "miss" and str(d.get("projectile_id", "")).startswith("proj_"):
            shooter = "_".join(str(d.get("projectile_id")).split("_")[1:3])
            if shooter in flankers:
                flank_misses += 1
        elif et == "torpedo_impact":
            victim = d.get("target_id")
            torp[role(victim)][d.get("hit_location")] += 1
            if victim in flankers:
                taken["attack" if attack_t is not None and e["timestamp"] >= attack_t else "position"] += 1
    out["shots_a"], out["shots_b"] = shots["A"], shots["B"]
    out["hits_on_a"] = dict(loc["A"]); out["hits_on_b"] = dict(loc["B"])
    out["pen_on_a"] = dict(pen["A"]); out["pen_on_b"] = dict(pen["B"])
    out["torp_on_a"] = dict(torp["A"]); out["torp_on_b"] = dict(torp["B"])
    out["flank_shots"] = flank_shots
    out["flank_hit_rate"] = flank_hits / (flank_hits + flank_misses) if flank_hits + flank_misses else None
    out["flank_hits_taken"] = taken
    out["flank_lost"] = {"position": sum(1 for s in flankers if s in dead and (attack_t is None or dead[s] < attack_t)),
                         "attack": sum(1 for s in flankers if s in dead and attack_t is not None and dead[s] >= attack_t)}
    fuel: Dict[str, float] = {}
    for e in events:
        if e["event_type"] == "execution_state" and e["ship_id"] in flankers:
            f = e["data"].get("fuel_fraction")
            if isinstance(f, (int, float)):
                fuel[e["ship_id"]] = f
    out["flank_fuel_used"] = (sum(1.0 - f for f in fuel.values()) / len(fuel)) if fuel else None

    # Geometry from the trace: time each flanker first reaches its station
    # bearing (|bearing - angle| <= 12 deg off the defender's facing axis);
    # the defender's facing (nose within 30 deg of a flanker vs of the main group).
    spec = spec_for(job["doc_a"], side_types(job, a_side), side_types(job, b_side))
    want_deg = spec.angle_deg if spec is not None else None
    reach: Dict[str, float] = {}
    face_f = face_m = samples = 0
    trace = rec.get("sim_trace") or []
    for frame in trace[::5]:
        ships = frame.get("ships", {})
        live = {k: s for k, s in ships.items() if not s.get("destroyed") and not s.get("dying")}
        b_guns = [s for k, s in live.items() if role(k) == "B"]
        a_main_live = [s for k, s in live.items() if k in a_main]
        if not b_guns:
            continue
        cb = [sum(s["pos"][i] for s in b_guns) / len(b_guns) for i in range(3)]
        if flankers and want_deg and a_main_live:
            ch = [sum(s["pos"][i] for s in a_main_live) / len(a_main_live) for i in range(3)]
            axis = [ch[i] - cb[i] for i in range(3)]
            na = math.sqrt(sum(v * v for v in axis)) or 1.0
            for k in flankers:
                if k in reach or k not in live:
                    continue
                v = [live[k]["pos"][i] - cb[i] for i in range(3)]
                nv = math.sqrt(sum(x * x for x in v)) or 1.0
                ang = math.degrees(math.acos(max(-1.0, min(1.0, sum(axis[i] * v[i] for i in range(3)) / (na * nv)))))
                if abs(ang - want_deg) <= 12.0:
                    reach[k] = frame["t"]
        if attack_t is None or frame["t"] < attack_t:
            continue
        fl = [s for k, s in live.items() if k in flankers]
        mn = [s for k, s in live.items() if role(k) == "A" and k not in flankers]
        for k, s in live.items():
            if role(k) != "B" or not s.get("fwd"):
                continue
            samples += 1

            def facing(group):
                for o in group:
                    v = [o["pos"][i] - s["pos"][i] for i in range(3)]
                    n = math.sqrt(sum(x * x for x in v)) or 1.0
                    if sum(s["fwd"][i] * v[i] for i in range(3)) / n >= NOSE_COS:
                        return True
                return False
            face_f += facing(fl)
            face_m += facing(mn)
    closest = None
    for frame in trace:
        ships = frame.get("ships", {})
        live = [(k, s) for k, s in ships.items() if not s.get("destroyed") and not s.get("dying")]
        for k, s in live:
            if role(k) != "A":
                continue
            for k2, s2 in live:
                if role(k2) == "B":
                    d = math.dist(s["pos"], s2["pos"])
                    if closest is None or d < closest:
                        closest = d
    out["closest_km"] = closest / 1000.0 if closest is not None else None
    out["flank_reach_s"] = {k: v for k, v in reach.items()}
    out["defender_face_flank"] = face_f / samples if samples else None
    out["defender_face_main"] = face_m / samples if samples else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True, help="docA:docB (A is the doctrine under test)")
    ap.add_argument("--rosters", nargs="+", default=list(AUTO_ROSTERS) + list(CUSTOM_ROSTERS))
    ap.add_argument("--budgets", type=int, nargs="+", default=[150, 200])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", action="store_true", help="only summarise the cache")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for pair in args.pairs:
        a, b = pair.split(":")
        for roster in args.rosters:
            for budget in args.budgets:
                types = roster_types(roster.split("/")[0], budget)
                for seed in args.seeds:
                    for a_is_alpha in (True, False):
                        tag = f"{a}-{b}_{roster.replace('/', '-vs-')}_b{budget}_s{seed}_{'A' if a_is_alpha else 'B'}alpha"
                        jobs.append(dict(tag=tag, out=str(out), roster=roster, budget=budget, types=types,
                                         seed=seed, a_is_alpha=a_is_alpha, doc_a=a, doc_b=b,
                                         alpha=a if a_is_alpha else b, beta=b if a_is_alpha else a,
                                         time_limit=args.time_limit))
    if args.summary:
        rows = [json.loads((out / f"{j['tag']}.json").read_text()) for j in jobs
                if (out / f"{j['tag']}.json").exists()]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(run_battle, jobs))
    (out / "_rows.json").write_text(json.dumps(rows))
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"{len(errors)} errors, e.g. {errors[0]['tag']}: {errors[0]['error']}")
    print_summary([r for r in rows if "error" not in r])
    return 0


def _pct(c: Dict[str, int]) -> str:
    n = sum(c.values())
    if not n:
        return "-"
    return "/".join(f"{100 * c.get(k, 0) / n:.0f}" for k in ("nose", "lateral", "tail")) + f" (n={n})"


def print_summary(rows: List[Dict[str, Any]]) -> None:
    by = defaultdict(list)
    for r in rows:
        by[(r["doc_a"], r["doc_b"], r["roster"])].append(r)
    print("| A vs B | roster | battles | A W/D/L | kills A-B | 1st kill s | hits on B n/l/t % | hits on A n/l/t % | pen/hit on B n/l/t | median closest km |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    totals = defaultdict(lambda: Counter())
    for key in sorted(by):
        rs = by[key]
        c = Counter(r["result"] for r in rs)
        hb, ha, pb = Counter(), Counter(), Counter()
        for r in rs:
            hb.update(r["hits_on_b"]); ha.update(r["hits_on_a"]); pb.update(r["pen_on_b"])
        fk = [r["first_kill_s"] for r in rs if r["first_kill_s"] is not None]
        penr = "/".join(f"{pb.get(k, 0) / hb[k]:.2f}" if hb.get(k) else "-" for k in ("nose", "lateral", "tail"))
        cl = sorted(r["closest_km"] for r in rs if r.get("closest_km") is not None)
        penr += f" | {cl[len(cl) // 2]:.0f}" if cl else " | -"
        print(f"| {key[0]} v {key[1]} | {key[2]} | {len(rs)} | {c['W']}/{c['D']}/{c['L']} | "
              f"{sum(r['kills_a'] for r in rs)}-{sum(r['kills_b'] for r in rs)} | "
              f"{sum(fk) / len(fk):.0f} | {_pct(hb)} | {_pct(ha)} | {penr} |" if fk else
              f"| {key[0]} v {key[1]} | {key[2]} | {len(rs)} | {c['W']}/{c['D']}/{c['L']} | "
              f"{sum(r['kills_a'] for r in rs)}-{sum(r['kills_b'] for r in rs)} | - | {_pct(hb)} | {_pct(ha)} | {penr} |")
        t = totals[(key[0], key[1])]
        t.update(c)
        t["kA"] += sum(r["kills_a"] for r in rs); t["kB"] += sum(r["kills_b"] for r in rs)
    print("\n| A vs B | total W/D/L | kills A-B |")
    print("|---|---|---|")
    for k, t in sorted(totals.items()):
        print(f"| {k[0]} v {k[1]} | {t['W']}/{t['D']}/{t['L']} | {t['kA']}-{t['kB']} |")
    # Flanker costs
    fr = [r for r in rows if r.get("n_flankers")]
    if fr:
        print("\n| A vs B | roster | flankers/battle | lost positioning | lost attack | hits taken positioning | attack | flank rounds | flank hit rate | fuel used | attack start s | defender nose on flank / main |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        g = defaultdict(list)
        for r in fr:
            g[(r["doc_a"], r["doc_b"], r["roster"])].append(r)
        for key in sorted(g):
            rs = g[key]
            n = len(rs)
            hr = [r["flank_hit_rate"] for r in rs if r["flank_hit_rate"] is not None]
            fu = [r["flank_fuel_used"] for r in rs if r["flank_fuel_used"] is not None]
            at = [r["attack_start_s"] for r in rs if r["attack_start_s"] is not None]
            ff = [r["defender_face_flank"] for r in rs if r["defender_face_flank"] is not None]
            fm = [r["defender_face_main"] for r in rs if r["defender_face_main"] is not None]
            mean = lambda v: sum(v) / len(v) if v else float("nan")   # noqa: E731
            print(f"| {key[0]} v {key[1]} | {key[2]} | {mean([r['n_flankers'] for r in rs]):.1f} | "
                  f"{sum(r['flank_lost']['position'] for r in rs)} | {sum(r['flank_lost']['attack'] for r in rs)} | "
                  f"{sum(r['flank_hits_taken']['position'] for r in rs)} | {sum(r['flank_hits_taken']['attack'] for r in rs)} | "
                  f"{sum(r['flank_shots'] for r in rs) / n:.0f} | {mean(hr):.0%} | {mean(fu):.2%} | {mean(at):.0f} | "
                  f"{mean(ff):.0%} / {mean(fm):.0%} |")


if __name__ == "__main__":
    sys.exit(main())
