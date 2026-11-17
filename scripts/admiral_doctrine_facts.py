#!/usr/bin/env python3
"""
Seeded facts behind the admiral doctrine's gun and torpedo numbers.

Two grids of heuristic-captain battles on the current engine, no LLM calls:

``guns``: gun fleets at 150 points, every unordered pair of four rosters
(gunline, bc_wall, combined, flagship), both heuristic doctrines (legacy,
tactical), seeds. From each recording:
  - per range band at firing (shot_fired.distance_km; a hit is matched to its
    shot by shooter, slot, target and firing time = hit time - flight time):
    rounds fired, hits, penetrations, hull GJ;
  - per target class x facing hit (hit_location nose / lateral / tail):
    hits, penetrations, hull GJ (also per range band, to see the confound).

``torpedo``: a five-corvette + three-frigate torpedo fleet (tactical
heuristic captains) against a 4 BC + 2 DD + 2 FF gun wall whose captains are
scripted (ScriptedWallCaptain):
  - static   : PADLOCK on the nearest attacker, no evade (the nose-on wall)
  - static_dodge : the same with evade "dodge"
  - open     : HEADING straight away from the attackers' centre, full burn
  - open_full: the same with evade "full"
  - charge   : INTERCEPT the nearest attacker (closure on purpose)
From each recording: rounds launched, impacts on the wall, blinded by PD,
dry / missed, impact speed, GJ landed, wall hulls lost and when, attackers lost.

    OMP_NUM_THREADS=1 uv run python scripts/admiral_doctrine_facts.py guns \\
        --seeds 1 2 3 --jobs 8 --out /tmp/facts
    OMP_NUM_THREADS=1 uv run python scripts/admiral_doctrine_facts.py torpedo \\
        --seeds 1 2 3 4 5 6 --jobs 8 --out /tmp/facts
    uv run python scripts/admiral_doctrine_facts.py guns --summary --out /tmp/facts

Each battle's metrics are cached as <out>/<grid>/<tag>.json (recordings go to
a temp dir and are deleted). The summary prints tables and writes
<out>/<grid>_summary.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BANDS_KM = (0, 25, 50, 100, 150, 200, 300, 10_000)
FACINGS = ("nose", "lateral", "tail")
GUN_ROSTERS = ("gunline", "bc_wall", "combined", "flagship")
CUSTOM_ROSTERS = {
    # What Sonnet 5.5 bought in three of four pass-5 battles (150 points).
    "bc_wall": ["battlecruiser"] * 4 + ["destroyer"] * 2 + ["frigate"] * 2,
    # Sonnet 5's torpedo fleet in battle b2 (141 points).
    "torpedo_boats": ["corvette"] * 5 + ["frigate"] * 3,
}
WALL_MODES = ("static", "static_dodge", "open", "open_full", "charge")


def band_of(km: float) -> str:
    for lo, hi in zip(BANDS_KM, BANDS_KM[1:]):
        if km < hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return f"{BANDS_KM[-2]}+"


def roster_types(name: str, budget: int = 150) -> List[str]:
    if name in CUSTOM_ROSTERS:
        return list(CUSTOM_ROSTERS[name])
    from src.llm.fleet_draft import auto_draft
    return [s.ship_type for s in auto_draft("alpha", budget=budget, seed=1, doctrine=name).ships]


# ---------------------------------------------------------------------------
# Scripted wall captain (torpedo grid)
# ---------------------------------------------------------------------------

def _wall_captain_class():
    from src.llm.heuristic_captain import HeuristicCaptain

    class ScriptedWallCaptain(HeuristicCaptain):
        """Flies one fixed posture: guns on the nearest attacker, plus the mode."""
        mode = "static"

        def _plan_tactical(self, ship, ship_id, enemies, simulation, checkpoint):
            calls = []
            target = min(enemies, key=lambda e: (e.position - ship.position).magnitude)
            calls.append(("set_primary_target", {"target_name": getattr(target, "name", target.ship_id)}))
            if self.weapon_groups and target.ship_id != self._last_target_id:
                calls.append(("set_weapons_order", {
                    "spinal_mode": "FIRE_WHEN_OPTIMAL", "turret_mode": "FIRE_WHEN_OPTIMAL",
                    "spinal_min_probability": 0.25, "turret_min_probability": 0.2}))
            self._last_target_id = target.ship_id
            mode = self.mode
            if mode.startswith("static"):
                args = {"maneuver_type": "PADLOCK", "target_id": target.ship_id}
                if mode == "static_dodge":
                    args["evade"] = "dodge"
                calls.append(("set_maneuver", args))
            elif mode.startswith("open"):
                n = len(enemies)
                cx = sum(e.position.x for e in enemies) / n
                cy = sum(e.position.y for e in enemies) / n
                cz = sum(e.position.z for e in enemies) / n
                away = (ship.position.x - cx, ship.position.y - cy, ship.position.z - cz)
                norm = sum(c * c for c in away) ** 0.5 or 1.0
                args = {"direction": {"x": away[0] / norm, "y": away[1] / norm, "z": away[2] / norm},
                        "throttle": 1.0}
                if mode == "open_full":
                    args["evade"] = "full"
                calls.append(("set_heading", args))
            elif mode == "charge":
                calls.append(("set_maneuver", {"maneuver_type": "INTERCEPT",
                                               "target_id": target.ship_id, "throttle": 1.0}))
            return calls

    return ScriptedWallCaptain


# ---------------------------------------------------------------------------
# Running one battle
# ---------------------------------------------------------------------------

def run_battle(job: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out_path = Path(job["out"]) / job["grid"] / f"{job['tag']}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    import contextlib
    import io
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_battle(job, out_path)


def _run_battle(job: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    from src.llm.fleet_config import BattleFleetConfig
    from src.llm.fleet_draft import FleetDraft, draft_to_fleet_definition, name_drafted_ships
    from src.llm.heuristic_captain import HEURISTIC_MODEL
    from src.llm.prompts import CaptainPersonality
    from src.recording_io import load_recording

    dist = float(job.get("distance_km", 500.0))
    drafts = {}
    for fac in ("alpha", "beta"):
        d = FleetDraft(faction=fac, budget=150, auto=True)
        d.ships = name_drafted_ships(job[f"{fac}_types"], fac)
        n = len(d.ships)
        for i, s in enumerate(d.ships):
            # The auto-draft "staggered wall" (fleet_draft.auto_draft).
            s.offset_km = (-10.0 if i % 2 else 0.0, (i - (n - 1) / 2) * 12.0, 6.0 if i % 3 == 2 else 0.0)
        drafts[fac] = d
    wall_mode = job.get("wall_mode")
    wall_cls = _wall_captain_class() if wall_mode else None

    class Runner(LLMBattleRunner):
        def _make_captain(self, captain_config, faction=None):
            if wall_cls is not None and faction == "beta":
                cap = wall_cls(captain_config, doctrine="tactical")
                cap.mode = wall_mode
                return cap
            return super()._make_captain(captain_config, faction)

    tmp = tempfile.mkdtemp(prefix="facts_")
    try:
        fc = BattleFleetConfig(
            battle_name=job["tag"], time_limit_s=job["time_limit"], initial_distance_km=dist,
            alpha_fleet=draft_to_fleet_definition(drafts["alpha"], dist, captain_model=HEURISTIC_MODEL),
            beta_fleet=draft_to_fleet_definition(drafts["beta"], dist, captain_model=HEURISTIC_MODEL),
            record_battle=True, record_sim_trace=False, personality_selection=False)
        bc = BattleConfig(initial_distance_km=dist, time_limit_s=job["time_limit"], max_checkpoints=200,
                          verbose=False, personality_selection=False, record_battle=True,
                          record_sim_trace=False, record_predictions=False, recording_dir=tmp,
                          seed=job["seed"],
                          heuristic_doctrine={"alpha": job["doctrine"], "beta": job["doctrine"]})
        dummy = LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                                 personality=CaptainPersonality.BALANCED)
        runner = Runner(config=bc, alpha_config=dummy, beta_config=dummy, client=None, fleet_config=fc)
        res = runner.run_battle(load_fleet_data())
        rec = load_recording(res.recording_file)
        m = analyze(rec, job)
        m["reason"] = res.reason
    except Exception as exc:  # recorded, never fatal to the sweep
        import traceback
        m = {"tag": job["tag"], "error": f"{type(exc).__name__}: {exc}",
             "trace": traceback.format_exc()[-1500:]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out_path.write_text(json.dumps(m))
    return m


def _ship_classes(rec: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for side in ("alpha", "beta"):
        fleet = rec.get(f"{side}_fleet") or {}
        for s in fleet.get("ships", []) if isinstance(fleet, dict) else []:
            sid = s.get("ship_id") or s.get("id")
            if sid:
                out[sid] = s.get("ship_type") or s.get("type") or "?"
    return out


def analyze(rec: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    events = rec.get("events", [])
    cls = _ship_classes(rec)
    if not cls:
        for e in events:
            if e["event_type"] == "battle_start":
                for s in (e.get("data") or {}).get("ships", []) or []:
                    cls[s.get("ship_id")] = s.get("ship_type")
    out: Dict[str, Any] = {"tag": job["tag"], "seed": job["seed"], "doctrine": job["doctrine"],
                           "grid": job["grid"], "winner": rec.get("winner") or "draw",
                           "duration_s": rec.get("duration_s")}
    # Shots, indexed for matching hits to their firing range.
    shots: Dict[tuple, List[tuple]] = defaultdict(list)
    band_shots: Dict[str, int] = defaultdict(int)
    for e in events:
        if e["event_type"] == "shot_fired":
            d = e["data"]
            km = float(d.get("distance_km") or 0.0)
            shots[(e["ship_id"], d.get("weapon_slot"), d.get("target_id"))].append((e["timestamp"], km))
            band_shots[band_of(km)] += 1
    band = defaultdict(lambda: {"shots": 0, "hits": 0, "pen": 0, "hull_gj": 0.0, "ke_gj": 0.0})
    for b, n in band_shots.items():
        band[b]["shots"] = n
    facing = defaultdict(lambda: {"hits": 0, "pen": 0, "hull_gj": 0.0, "ke_gj": 0.0})
    unmatched = 0
    for e in events:
        if e["event_type"] != "hit":
            continue
        d = e["data"]
        if not str(d.get("projectile_id", "proj_")).startswith("proj_"):
            continue
        victim = e["ship_id"]
        fired_t = e["timestamp"] - float(d.get("flight_time_s") or 0.0)
        cands = shots.get((d.get("shooter_id"), d.get("weapon_slot"), victim)) or []
        best = min(cands, key=lambda c: abs(c[0] - fired_t), default=None)
        if best is None or abs(best[0] - fired_t) > 2.0:
            unmatched += 1
            km = None
        else:
            km = best[1]
        loc = d.get("hit_location") or "?"
        pen = 1 if d.get("penetrated") else 0
        gj = float(d.get("damage_to_hull_gj") or 0.0)
        ke = float(d.get("kinetic_energy_gj") or 0.0)
        tcls = cls.get(victim, "?")
        for key in ((tcls, loc, "all"), ("ALL", loc, "all")) + (
                ((tcls, loc, band_of(km)), ("ALL", loc, band_of(km))) if km is not None else ()):
            f = facing["|".join(key)]
            f["hits"] += 1
            f["pen"] += pen
            f["hull_gj"] += gj
            f["ke_gj"] += ke
        if km is not None:
            b = band[band_of(km)]
            b["hits"] += 1
            b["pen"] += pen
            b["hull_gj"] += gj
            b["ke_gj"] += ke
    out["bands"] = dict(band)
    out["facing"] = dict(facing)
    out["unmatched_hits"] = unmatched

    dead: Dict[str, float] = {}
    for e in events:
        if e["event_type"] in ("ship_dying", "ship_destroyed"):
            dead.setdefault(e["ship_id"], e["timestamp"])
    out["dead"] = dead
    if job["grid"] == "torpedo":
        t = {"launched": 0, "impacts": 0, "impact_gj": 0.0, "impact_kps": [], "intercepted": 0,
             "pd_disabled": 0, "dry": 0, "missed": 0, "impact_loc": defaultdict(int),
             "first_launch_s": None, "launch_km": []}
        for e in events:
            et, d = e["event_type"], e.get("data") or {}
            if et == "torpedo_launched":
                t["launched"] += 1
                if t["first_launch_s"] is None:
                    t["first_launch_s"] = e["timestamp"]
                if d.get("launch_distance_km") is not None:
                    t["launch_km"].append(float(d["launch_distance_km"]))
            elif et == "torpedo_impact":
                t["impacts"] += 1
                t["impact_gj"] += float(d.get("total_damage_gj") or 0.0)
                t["impact_kps"].append(float(d.get("impact_speed_kps") or 0.0))
                t["impact_loc"][d.get("hit_location") or "?"] += 1
            elif et == "torpedo_intercepted":
                t["intercepted"] += 1
            elif et == "pd_torpedo_disabled":
                t["pd_disabled"] += 1
            elif et == "torpedo_fuel_exhausted":
                t["dry"] += 1
            elif et == "torpedo_miss":
                t["missed"] += 1
        t["impact_loc"] = dict(t["impact_loc"])
        out["torpedo"] = t
        out["wall_mode"] = job["wall_mode"]
        out["attacker"] = job.get("attacker", "torpedo_boats")
        out["distance_km"] = job.get("distance_km", 500.0)
        out["wall_lost"] = sorted(v for k, v in dead.items() if k.startswith("beta"))
        out["boats_lost"] = sorted(v for k, v in dead.items() if k.startswith("alpha"))
        out["wall_n"] = len(job["beta_types"])
        out["boats_n"] = len(job["alpha_types"])
    return out


# ---------------------------------------------------------------------------
# Grids and summaries
# ---------------------------------------------------------------------------

def gun_jobs(args) -> List[Dict[str, Any]]:
    jobs = []
    for a, b in itertools.combinations_with_replacement(GUN_ROSTERS, 2):
        for doctrine in ("legacy", "tactical"):
            for seed in args.seeds:
                tag = f"{a}-{b}_{doctrine}_s{seed}"
                jobs.append(dict(grid="guns", tag=tag, out=args.out, seed=seed, doctrine=doctrine,
                                 alpha_types=roster_types(a), beta_types=roster_types(b),
                                 time_limit=args.time_limit))
    return jobs


def torpedo_jobs(args) -> List[Dict[str, Any]]:
    jobs = []
    for attacker in args.attackers:
        for dist in args.distances:
            for mode in WALL_MODES:
                for seed in args.seeds:
                    tag = f"wall_{mode}_{attacker}_d{dist:.0f}_s{seed}"
                    jobs.append(dict(grid="torpedo", tag=tag, out=args.out, seed=seed,
                                     doctrine="tactical", wall_mode=mode, attacker=attacker,
                                     distance_km=dist, alpha_types=roster_types(attacker),
                                     beta_types=roster_types("bc_wall"), time_limit=args.time_limit))
    return jobs


def _pct(n: float, d: float) -> str:
    return f"{100.0 * n / d:.0f}%" if d else "-"


def summarize_guns(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"battles": len(rows), "by_doctrine": {}}
    for doctrine in ("legacy", "tactical", "both"):
        sel = [r for r in rows if doctrine == "both" or r["doctrine"] == doctrine]
        bands = defaultdict(lambda: defaultdict(float))
        facing = defaultdict(lambda: defaultdict(float))
        for r in sel:
            for b, v in r["bands"].items():
                for k, x in v.items():
                    bands[b][k] += x
            for f, v in r["facing"].items():
                for k, x in v.items():
                    facing[f][k] += x
        summary["by_doctrine"][doctrine] = {"battles": len(sel),
                                            "bands": {k: dict(v) for k, v in bands.items()},
                                            "facing": {k: dict(v) for k, v in facing.items()}}
        order = [band_of(lo) for lo in BANDS_KM[:-1]]
        total_gj = sum(bands[b]["hull_gj"] for b in order) or 1.0
        print(f"\n== guns, doctrine {doctrine}: {len(sel)} battles ==")
        print(f"{'band km':>9} {'shots':>6} {'hits':>5} {'hit%':>5} {'pen%':>5} {'GJ/hit':>7} {'hullGJ':>7} {'share':>6}")
        for b in order:
            v = bands[b]
            print(f"{b:>9} {int(v['shots']):6d} {int(v['hits']):5d} {_pct(v['hits'], v['shots']):>5} "
                  f"{_pct(v['pen'], v['hits']):>5} {v['hull_gj'] / v['hits'] if v['hits'] else 0:7.2f} "
                  f"{v['hull_gj']:7.1f} {_pct(v['hull_gj'], total_gj):>6}")
        print(f"{'target':>14} {'facing':>8} {'hits':>5} {'pen%':>5} {'hullGJ/hit':>10} {'KE/hit':>7}")
        classes = sorted({k.split("|")[0] for k in facing})
        for c in classes:
            for f in FACINGS:
                v = facing.get(f"{c}|{f}|all")
                if not v or not v["hits"]:
                    continue
                print(f"{c:>14} {f:>8} {int(v['hits']):5d} {_pct(v['pen'], v['hits']):>5} "
                      f"{v['hull_gj'] / v['hits']:10.3f} {v['ke_gj'] / v['hits']:7.2f}")
        print("ALL targets, facing x band (hits / pen% / hull GJ per hit):")
        for f in FACINGS:
            cells = []
            for b in order:
                v = facing.get(f"ALL|{f}|{b}")
                cells.append(f"{b}:{int(v['hits'])}/{_pct(v['pen'], v['hits'])}/{v['hull_gj'] / v['hits']:.2f}"
                             if v and v["hits"] else f"{b}:-")
            print(f"  {f:>8} " + "  ".join(cells))
    return summary


def summarize_torpedo(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"battles": len(rows), "modes": {}}
    keys = sorted({(r.get("attacker", "torpedo_boats"), float(r.get("distance_km", 500.0))) for r in rows})
    for attacker, dist in keys:
        print(f"\n== torpedo strike ({attacker}, tactical) vs 4 BC + 2 DD + 2 FF wall, start {dist:.0f} km ==")
        _summarize_torpedo_cell([r for r in rows if r.get("attacker", "torpedo_boats") == attacker
                                 and float(r.get("distance_km", 500.0)) == dist],
                                summary, f"{attacker}|{dist:.0f}")
    return summary


def _summarize_torpedo_cell(rows, summary, prefix):
    print(f"{'mode':>13} {'n':>2} {'launched':>8} {'impacts':>7} {'hit%':>5} {'kps med':>7} "
          f"{'GJ/impact':>9} {'wall lost':>9} {'boats lost':>10} {'wall win':>8} {'first loss s':>12}")
    for mode in WALL_MODES:
        sel = [r for r in rows if r.get("wall_mode") == mode]
        if not sel:
            continue
        launched = sum(r["torpedo"]["launched"] for r in sel)
        impacts = sum(r["torpedo"]["impacts"] for r in sel)
        gj = sum(r["torpedo"]["impact_gj"] for r in sel)
        speeds = sorted(s for r in sel for s in r["torpedo"]["impact_kps"])
        med = speeds[len(speeds) // 2] if speeds else 0.0
        wall_lost = sum(len(r["wall_lost"]) for r in sel)
        boats_lost = sum(len(r["boats_lost"]) for r in sel)
        wins = sum(1 for r in sel if r["winner"] == "beta")
        firsts = sorted(r["wall_lost"][0] for r in sel if r["wall_lost"])
        first = firsts[len(firsts) // 2] if firsts else None
        locs = defaultdict(int)
        for r in sel:
            for k, v in r["torpedo"]["impact_loc"].items():
                locs[k] += v
        key = f"{prefix}|{mode}"
        summary["modes"][key] = dict(
            battles=len(sel), launched=launched, impacts=impacts, impact_gj=gj,
            impact_kps_median=med, impact_kps=[speeds[0], speeds[-1]] if speeds else None,
            wall_lost=wall_lost, wall_total=sum(r["wall_n"] for r in sel), boats_lost=boats_lost,
            boats_total=sum(r["boats_n"] for r in sel), wall_wins=wins, median_first_loss_s=first,
            impact_loc=dict(locs), intercepted=sum(r["torpedo"]["intercepted"] for r in sel),
            pd_disabled=sum(r["torpedo"]["pd_disabled"] for r in sel),
            dry=sum(r["torpedo"]["dry"] for r in sel), missed=sum(r["torpedo"]["missed"] for r in sel))
        print(f"{mode:>13} {len(sel):2d} {launched:8d} {impacts:7d} {_pct(impacts, launched):>5} {med:7.1f} "
              f"{gj / impacts if impacts else 0:9.1f} {wall_lost:4d}/{sum(r['wall_n'] for r in sel):<4d} "
              f"{boats_lost:5d}/{sum(r['boats_n'] for r in sel):<4d} {wins:4d}/{len(sel):<3d} "
              f"{first if first is not None else '-':>12}")
        m = summary["modes"][key]
        print(f"{'':>13} impacts by facing {dict(locs)}; PD disabled {m['pd_disabled']}, "
              f"dry {m['dry']}, missed {m['missed']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", choices=("guns", "torpedo"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--distances", type=float, nargs="+", default=[500.0],
                    help="torpedo grid: start separations in km")
    ap.add_argument("--attackers", nargs="+", default=["torpedo_boats"],
                    help="torpedo grid: attacking rosters (torpedo_boats, torpedo = auto-draft)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", action="store_true", help="only summarise the cache")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = gun_jobs(args) if args.grid == "guns" else torpedo_jobs(args)
    if args.summary:
        rows = [json.loads(p.read_text()) for p in (Path(args.out) / args.grid).glob("*.json")]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(run_battle, jobs))
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"{len(errors)} errors, e.g. {errors[0]['tag']}: {errors[0]['error']}\n{errors[0].get('trace', '')}")
    rows = [r for r in rows if "error" not in r]
    summary = summarize_guns(rows) if args.grid == "guns" else summarize_torpedo(rows)
    summary["command"] = " ".join(sys.argv)
    summary["seeds"] = args.seeds
    (Path(args.out) / f"{args.grid}_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
