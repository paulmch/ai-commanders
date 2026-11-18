#!/usr/bin/env python3
"""
Torpedo retarget sweep: what happens to rounds whose target dies in flight,
over seeded heuristic battles (rule-based captains, no admirals, no LLM),
and what candidate retarget rules would change.

Every battle is instrumented (inert wrappers, see --check-inert):
- CombatSimulation.step: after each step, every round whose target is gone
  (destroyed) is an orphan; its seeker / tank state at the first sighting is
  kept; every round homing on a DYING hulk is noted.
- CombatSimulation._retarget_torpedo: before the engine's own call, every
  candidate's gate from the engine's pure can_intercept (no_solution / dv /
  time / reserve / ok, engine order) - the binding gate of an attempt is the
  best one any candidate reached.
- CombatSimulation._resolve_torpedo_hit: whether the struck hull was a dying
  hulk, a friendly, the round's launch target or its retarget.

Options (counterfactual rules, applied by runtime patches inside the worker
process only; src/ is never edited):
    baseline    the engine as it is
    reserve0    retarget reserve 0 (RETARGET_MIN_RESERVE_KPS = FRACTION = 0)
    scaled      reserve min(1.12, max(0.1, 0.5 * a_max(candidate) * t_intercept))
    floorgate   the gate also charges the cruise-floor closure burn:
                fuel_left - max(0, MIN_CLOSING_SPEED_KPS - closure)
    nofloor     a retargeted round's guidance has no cruise floor (it spends
                its tank on the steer, not on regaining 12 km/s closure)
    nofloor_r0  nofloor + reserve0
    dying       retarget the moment the target becomes a dying hulk (if a
                live enemy is reachable; else keep homing on the hulk)
    dying_nofloor  dying + nofloor

    uv run python scripts/torpedo_retarget_sweep.py --options baseline,reserve0 \
        --seeds 3 --workers 6 --out /path/sweep.jsonl
    uv run python scripts/torpedo_retarget_sweep.py --out /path/sweep.jsonl --summary-only
    uv run python scripts/torpedo_retarget_sweep.py --check-inert
"""
from __future__ import annotations

import os
import sys

_PINNED_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
if __name__ == "__main__" and any(os.environ.get(k) != v for k, v in _PINNED_ENV.items()):
    os.environ.update(_PINNED_ENV)
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import math
import multiprocessing as mp
import signal
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_torpedo_cost as calib  # noqa: E402

# Torpedo-armed rosters against gun rosters (150-point draft_tournament
# rosters at the checked-in prices) plus the p5w5_b3 line-up.
FLEETS = {
    "torpedo": ["cruiser_torpedo", "corvette", "corvette"],
    "swarm": ["corvette"] * 6,
    "screen": ["cruiser_torpedo"] + ["frigate"] * 7,
    "b3_alpha": ["corvette"] * 5 + ["frigate"] * 3,
    "gunline": ["destroyer"] * 4 + ["cruiser"] * 2 + ["frigate"] * 2,
    "battleline": ["battleship"] * 3 + ["destroyer"],
    "escort": ["battleship"] * 2 + ["cruiser"] + ["frigate"] * 5,
    "b3_beta": ["battlecruiser"] * 5 + ["frigate"] * 3,
}
DEFAULT_PAIRS = [("torpedo", "gunline"), ("torpedo", "battleline"),
                 ("swarm", "gunline"), ("swarm", "escort"),
                 ("screen", "battleline"), ("b3_alpha", "b3_beta"),
                 ("torpedo", "swarm")]
OPTIONS = ["baseline", "reserve0", "scaled", "floorgate", "nofloor",
           "nofloor_r0", "dying", "dying_nofloor"]
# The engine as built (pass 5): no link (the pre-link rule, = baseline),
# sprint with the datalink, every launch on the reserve profile with the link.
BUILT = {"p5_nolink": {"AIC_TORPEDO_DATALINK": "0"},
         "p5_link": {"AIC_TORPEDO_DATALINK": "1"},
         "p5_reserve": {"AIC_TORPEDO_DATALINK": "1", "AIC_TORPEDO_PROFILE": "reserve"}}


# ---------------------------------------------------------------------------
# Option patches (worker process only)
# ---------------------------------------------------------------------------

def apply_option(option: str) -> None:
    if option in BUILT:
        os.environ.update(BUILT[option])
        return
    # the investigation's counterfactuals patch the PRE-link per-round rule
    os.environ["AIC_TORPEDO_DATALINK"] = "0"
    from src import simulation as S
    from src import torpedo as T
    CS = S.CombatSimulation
    parts = set(option.split("_")) if option != "dying_nofloor" else {"dying", "nofloor"}
    if option == "nofloor_r0":
        parts = {"nofloor", "reserve0"}

    if "reserve0" in parts:
        CS.RETARGET_MIN_RESERVE_KPS = 0.0
        CS.RETARGET_RESERVE_FRACTION = 0.0

    if option in ("scaled", "floorgate"):
        engine_retarget = CS._retarget_torpedo

        def _retarget(self, tf):
            torp = tf.torpedo
            reserve_full = max(self.RETARGET_MIN_RESERVE_KPS,
                               torp.specs.total_delta_v_kps * self.RETARGET_RESERVE_FRACTION)
            heat = tf.heat_absorbed_j / tf.ELECTRONICS_THRESHOLD_J
            hurried = heat >= self.RETARGET_HURRY_HEAT_FRACTION
            best, best_score = None, -float("inf")
            for c in self.get_enemy_ships(tf.source_ship_id):
                ok, t, left = torp.can_intercept(c.position, c.velocity)
                if not ok or t > self.RETARGET_MAX_INTERCEPT_S:
                    continue
                if option == "scaled":
                    a = c.kinematic_state.max_acceleration_g() * 9.81
                    reserve = min(reserve_full, max(0.1, 0.5 * a * t / 1000.0))
                else:
                    rel = c.position - torp.position
                    clos = -(c.velocity - torp.velocity).dot(rel.normalized()) / 1000.0
                    left -= max(0.0, T.MIN_CLOSING_SPEED_KPS - clos)
                    reserve = reserve_full
                if left < reserve:
                    continue
                score = -t if hurried else left
                if score > best_score:
                    best, best_score = c, score
            if best is None:
                return None
            # let the engine do the switch + event, restricted to this pick
            try:
                self.get_enemy_ships = lambda _sid: [best]
                self.RETARGET_MIN_RESERVE_KPS = 0.0
                self.RETARGET_RESERVE_FRACTION = 0.0
                return engine_retarget(self, tf)
            finally:
                del self.get_enemy_ships
                del self.RETARGET_MIN_RESERVE_KPS
                del self.RETARGET_RESERVE_FRACTION
        CS._retarget_torpedo = _retarget

    if "nofloor" in parts:
        inner = CS._retarget_torpedo

        def _retarget_mark(self, tf):
            new = inner(self, tf)
            if new is not None:
                tf.torpedo._aic_retargeted = True
            return new
        CS._retarget_torpedo = _retarget_mark
        guide = T.TorpedoGuidance._collision_course_guidance

        def _guidance(self, torpedo, *a, **kw):
            if not getattr(torpedo, "_aic_retargeted", False):
                return guide(self, torpedo, *a, **kw)
            saved = T.MIN_CLOSING_SPEED_KPS
            T.MIN_CLOSING_SPEED_KPS = 0.0
            try:
                return guide(self, torpedo, *a, **kw)
            finally:
                T.MIN_CLOSING_SPEED_KPS = saved
        T.TorpedoGuidance._collision_course_guidance = _guidance

    if "dying" in parts:
        upd = CS._update_torpedoes

        def _update(self, dt):
            for tf in self.torpedoes:
                tgt = self.get_ship(tf.torpedo.target_id)
                if (tgt is not None and tgt.is_dying and not tgt.is_destroyed
                        and not tf.is_disabled and not tf.torpedo.fuel_exhausted):
                    self._retarget_torpedo(tf)
            return upd(self, dt)
        CS._update_torpedoes = _update


# ---------------------------------------------------------------------------
# Instrumentation (inert)
# ---------------------------------------------------------------------------

GATE_ORDER = ("ok", "reserve", "dv", "time", "no_solution")


def instrument(log: Dict[str, Any]) -> None:
    from src import simulation as S
    CS = S.CombatSimulation
    rounds = log["rounds"]

    def rec(tid):
        return rounds.setdefault(tid, {"attempts": 0, "gates": [], "retargets": [],
                                       "orphan": None, "hulk_homing": False})

    step = CS.step

    def _step(self):
        out = step(self)
        for tf in self.torpedoes:
            tgt = self.get_ship(tf.torpedo.target_id)
            r = rec(tf.torpedo_id)
            if tgt is None or tgt.is_destroyed:
                if r["orphan"] is None or r["orphan"]["target"] != tf.torpedo.target_id:
                    r["orphan"] = {"target": tf.torpedo.target_id, "t": self.current_time,
                                   "disabled": tf.is_disabled,
                                   "dry": tf.torpedo.fuel_exhausted,
                                   "dv": round(tf.torpedo.remaining_delta_v_kps, 3)}
                    r.setdefault("orphan_episodes", []).append(r["orphan"])
            elif tgt.is_dying:
                r["hulk_homing"] = True
        return out
    CS.step = _step

    retarget = CS._retarget_torpedo

    def _retarget(self, tf):
        torp = tf.torpedo
        reserve = max(self.RETARGET_MIN_RESERVE_KPS,
                      torp.specs.total_delta_v_kps * self.RETARGET_RESERVE_FRACTION)
        best = None
        for c in self.get_enemy_ships(tf.source_ship_id):
            ok, t, left = torp.can_intercept(c.position, c.velocity)
            if math.isinf(t):
                g = "no_solution"
            elif not ok:
                g = "dv"
            elif t > self.RETARGET_MAX_INTERCEPT_S:
                g = "time"
            elif left < reserve:
                g = "reserve"
            else:
                g = "ok"
            if best is None or GATE_ORDER.index(g) < GATE_ORDER.index(best):
                best = g
        r = rec(tf.torpedo_id)
        r["attempts"] += 1
        if not r["gates"] or r["gates"][-1] != best:
            r["gates"].append(best or "no_candidates")
        new = retarget(self, tf)
        if new is not None:
            r["retargets"].append({"t": self.current_time, "to": new.ship_id,
                                   "dv": round(torp.remaining_delta_v_kps, 3)})
        return new
    CS._retarget_torpedo = _retarget

    hit = CS._resolve_torpedo_hit

    def _hit(self, tf, target):
        src = self.get_ship(tf.source_ship_id)
        r = rec(tf.torpedo_id)
        r["impact"] = {"t": self.current_time, "ship": target.ship_id,
                       "dying": bool(target.is_dying),
                       "friendly": bool(src is not None and src.faction == target.faction),
                       "is_target": target.ship_id == tf.torpedo.target_id,
                       "guided": not tf.is_disabled and not (
                           self.get_ship(tf.torpedo.target_id) is None
                           or self.get_ship(tf.torpedo.target_id).is_destroyed)}
        return hit(self, tf, target)
    CS._resolve_torpedo_hit = _hit


# ---------------------------------------------------------------------------
# One battle
# ---------------------------------------------------------------------------

def run_battle(task: Dict[str, Any]) -> Dict[str, Any]:
    if not task.get("verbose"):
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    signal.signal(signal.SIGALRM, calib._timeout)
    signal.alarm(int(task.get("battle_timeout", 3 * 3600)))
    t0 = time.time()
    out = dict(task)
    try:
        calib.load_engine(ROOT)
        apply_option(task["option"]) if task["option"] != "baseline" else \
            os.environ.update({"AIC_TORPEDO_DATALINK": "0"})
        log: Dict[str, Any] = {"rounds": {}}
        if task.get("instrument", True):
            instrument(log)
        E = calib.ENGINE
        fd = E["fleet_draft"]
        drafts = {}
        for faction in ("alpha", "beta"):
            roster = task[f"{faction}_roster"]
            draft = fd.FleetDraft(faction=faction, budget=0, auto=True,
                                  formation_name="staggered wall")
            draft.ships = fd.name_drafted_ships(list(roster), faction)
            for ship, off in zip(draft.ships, calib.formation_offsets("staggered", roster, {})):
                ship.offset_km = tuple(off)
            drafts[faction] = draft
        kw = calib._dataclass_kwargs
        dist = task["distance_km"]
        fleet_cfg = E["BattleFleetConfig"](**kw(
            E["BattleFleetConfig"], battle_name="retarget sweep",
            alpha_fleet=fd.draft_to_fleet_definition(drafts["alpha"], dist,
                                                     captain_model=E["HEURISTIC_MODEL"]),
            beta_fleet=fd.draft_to_fleet_definition(drafts["beta"], dist,
                                                    captain_model=E["HEURISTIC_MODEL"]),
            time_limit_s=task["time_limit_s"], initial_distance_km=dist,
            record_battle=False, record_sim_trace=False,
            personality_selection=False, use_notebooks=False))
        battle_cfg = E["BattleConfig"](**kw(
            E["BattleConfig"], initial_distance_km=dist, time_limit_s=task["time_limit_s"],
            max_checkpoints=20, verbose=False, personality_selection=False,
            record_battle=False, record_sim_trace=False, record_predictions=False,
            seed=task["seed"]))
        dummy = E["LLMCaptainConfig"](name="-", ship_name="-", model=E["HEURISTIC_MODEL"],
                                      personality=E["CaptainPersonality"].BALANCED)
        runner = E["LLMBattleRunner"](config=battle_cfg, alpha_config=dummy,
                                      beta_config=dummy, client=None, fleet_config=fleet_cfg)
        result = runner.run_battle(E["fleet_data"])
        sim = runner.simulation
        ET = E["EventType"]
        launched, fate = {}, {}
        rt_events = defaultdict(list)
        for e in sim.events:
            d = e.data or {}
            if e.event_type == ET.TORPEDO_RETARGETED:
                rt_events[d["torpedo_id"]].append({"t": e.timestamp, "to": d["new_target_id"],
                                                   "reason": d.get("reason")})
            if e.event_type == ET.TORPEDO_LAUNCHED:
                launched[d["torpedo_id"]] = {"src": e.ship_id, "target": d.get("target_id")
                                             or e.target_id, "t": e.timestamp}
            elif e.event_type == ET.TORPEDO_IMPACT:
                fate[d["torpedo_id"]] = "impact"
            elif e.event_type == ET.PROJECTILE_MISS and d.get("type") == "torpedo":
                fate[d["projectile_id"]] = "miss_" + (d.get("reason") or "cpa")
            elif e.event_type in (ET.TORPEDO_INTERCEPTED, ET.PD_TORPEDO_DESTROYED):
                fate.setdefault(d.get("torpedo_id"), "destroyed")
        for tf in sim.torpedoes:
            fate.setdefault(tf.torpedo_id, "in_flight_at_end")
        deaths = {}
        for e in sim.events:
            if e.event_type in (ET.SHIP_DYING, ET.SHIP_DESTROYED) and e.ship_id not in deaths:
                deaths[e.ship_id] = e.timestamp
        rounds = []
        for tid, info in launched.items():
            r = log["rounds"].get(tid, {})
            rounds.append({"id": tid, "side": info["src"].split("_")[0], **info,
                           "fate": fate.get(tid, "unknown"),
                           "orphan_episodes": r.get("orphan_episodes", []),
                           "attempts": r.get("attempts", 0), "gates": r.get("gates", []),
                           "retargets": rt_events.get(tid, []),
                           "hulk_homing": r.get("hulk_homing", False),
                           "impact": r.get("impact")})
        ships = {**runner.alpha_ships, **runner.beta_ships}
        lost = {f: sum(1 for s in fl.values() if s.is_destroyed or s.is_dying
                       or s.is_surrendered)
                for f, fl in (("alpha", runner.alpha_ships), ("beta", runner.beta_ships))}
        out.update(winner=result.winner, reason=result.reason,
                   duration_s=round(result.duration_s, 1), lost=lost, rounds=rounds,
                   deaths=deaths,
                   n_events=len(sim.events), n_ships=len(ships), error=None)
    except BaseException as exc:
        out.update(winner=None, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-2000:])
    finally:
        signal.alarm(0)
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_opt: Dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if r.get("error"):
            by_opt[r["option"]]["errors"] += 1
            continue
        c = by_opt[r["option"]]
        c["battles"] += 1
        torp_sides = {s for s in ("alpha", "beta")
                      if any(x["side"] == s for x in r["rounds"])}
        gun_side = [s for s in ("alpha", "beta") if s not in torp_sides]
        if len(torp_sides) == 1 and gun_side:
            ts = next(iter(torp_sides))
            c["torp_vs_gun_battles"] += 1
            c["torp_side_wins"] += r["winner"] == ts
            c["gun_side_wins"] += r["winner"] == gun_side[0]
            c["torp_side_lost_ships"] += r["lost"][ts]
            c["gun_side_lost_ships"] += r["lost"][gun_side[0]]
        if len(torp_sides) == 1 and gun_side:
            c["torp_side_rounds_fired"] += sum(1 for x in r["rounds"] if x["side"] == ts)
            c["torp_side_rounds_hit"] += sum(1 for x in r["rounds"]
                                             if x["side"] == ts and x.get("impact"))
            c["torp_vs_gun_battle_s"] += r["duration_s"]
            for sid, t in (r.get("deaths") or {}).items():
                if sid.startswith(gun_side[0]):
                    for cut in (150, 200, 300):
                        c[f"gun_side_lost_by_{cut}s"] += t <= cut
        for x in r["rounds"]:
            c["launched"] += 1
            c[f"fate_{x['fate']}"] += 1
            imp = x.get("impact")
            if imp:
                c["impacts"] += 1
                c["impact_on_hulk"] += imp["dying"]
                c["impact_on_friendly"] += imp["friendly"]
            if x["hulk_homing"]:
                c["homed_on_hulk"] += 1
                if imp and imp["dying"]:
                    c["homed_on_hulk_hit_hulk"] += 1
            eps = x["orphan_episodes"]
            if any(rt.get("reason") == "datalink_surplus" for rt in x["retargets"]):
                c["diverted_surplus"] += 1
                if imp and imp["ship"] in [rt["to"] for rt in x["retargets"]]:
                    c["diverted_surplus_hit"] += 1
            lost_rts = [rt for rt in x["retargets"] if rt.get("reason") != "datalink_surplus"]
            orphaned = bool(eps) or bool(lost_rts)
            if not orphaned:
                continue
            c["orphaned"] += 1
            first = eps[0] if eps else None
            if lost_rts:
                c["retargeted"] += 1
                c["retargeted_multi"] += len(lost_rts) > 1
                if imp and imp["ship"] in [rt["to"] for rt in lost_rts]:
                    c["retargeted_hit_new_target"] += 1
                elif imp:
                    c["retargeted_hit_other"] += 1
                else:
                    c[f"retargeted_then_{x['fate']}"] += 1
            elif first and first["disabled"]:
                c["orphan_blind"] += 1
            elif first and first["dry"]:
                c["orphan_dry"] += 1
            else:
                g = x["gates"][0] if x["gates"] else "no_attempt"
                c[f"orphan_live_refused_{g}"] += 1
                if imp:
                    c["orphan_live_refused_but_hit"] += 1
            if x["hulk_homing"] and first and not x["retargets"]:
                c["orphaned_after_hulk"] += 1
    return {k: dict(v) for k, v in by_opt.items()}


def print_summary(s: Dict[str, Any]) -> None:
    keys = sorted({k for v in s.values() for k in v})
    opts = [o for o in OPTIONS if o in s] + [o for o in s if o not in OPTIONS]
    w = max(len(k) for k in keys) + 2
    print("".join([" " * w] + [f"{o:>14s}" for o in opts]))
    for k in keys:
        print(f"{k:<{w}s}" + "".join(f"{s[o].get(k, 0):>14}" for o in opts))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--options", type=lambda x: x.split(","), default=OPTIONS)
    ap.add_argument("--pairs", type=lambda x: [tuple(p.split(":")) for p in x.split(",")],
                    default=DEFAULT_PAIRS, help="a:b,c:d (FLEETS names)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--distance", type=float, default=500.0)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None, help="JSONL cache")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--check-inert", action="store_true",
                    help="one battle with and without instrumentation: identical?")
    args = ap.parse_args()

    if args.check_inert:
        base = dict(option="baseline", seed=1, alpha_roster=FLEETS["b3_alpha"],
                    beta_roster=FLEETS["b3_beta"], distance_km=args.distance,
                    time_limit_s=300.0)
        with mp.get_context("fork").Pool(2, maxtasksperchild=1) as pool:
            a, b = pool.map(run_battle, [dict(base, instrument=True),
                                         dict(base, instrument=False)])
        same = all(a[k] == b[k] for k in ("winner", "reason", "duration_s", "lost", "n_events"))
        fa = sorted((x["id"], x["fate"]) for x in a["rounds"])
        fb = sorted((x["id"], x["fate"]) for x in b["rounds"])
        print("instrumented == plain:", same and fa == fb,
              {k: (a[k], b[k]) for k in ("winner", "reason", "n_events", "lost")})
        return 0 if same and fa == fb else 1

    cache: Dict[str, Dict[str, Any]] = {}
    if args.out and args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["key"]] = r
    tasks = []
    for opt in args.options:
        for a, b in args.pairs:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                for flip in (False, True):
                    al, be = (b, a) if flip else (a, b)
                    key = f"{opt}|{al}|{be}|{seed}|{args.distance}|{args.time_limit}"
                    if key in cache:
                        continue
                    tasks.append(dict(key=key, option=opt, seed=seed, alpha=al, beta=be,
                                      alpha_roster=FLEETS[al], beta_roster=FLEETS[be],
                                      distance_km=args.distance,
                                      time_limit_s=args.time_limit))
    if tasks and not args.summary_only:
        print(f"{len(tasks)} battles to fight ({len(cache)} cached)", flush=True)
        t0 = time.time()
        with mp.get_context("fork").Pool(args.workers, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(run_battle, tasks):
                cache[r["key"]] = r
                if args.out:
                    with args.out.open("a") as f:
                        f.write(json.dumps(r) + "\n")
        print(f"fought in {time.time() - t0:.0f}s", flush=True)
    wanted = {f"{opt}|{al}|{be}|{seed}|{args.distance}|{args.time_limit}"
              for opt in args.options for a, b in args.pairs
              for seed in range(args.seed_start, args.seed_start + args.seeds)
              for al, be in ((a, b), (b, a))}
    results = [r for k, r in cache.items() if k in wanted]
    errs = [r for r in results if r.get("error")]
    for r in errs[:3]:
        print("ERROR", r["key"], r["error"], r.get("traceback", "")[-600:])
    print_summary(summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
