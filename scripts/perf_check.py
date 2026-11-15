#!/usr/bin/env python3
"""
Engine performance + bit-equality check.

Re-runs a fixed set of seeded, free (no-LLM) battles and a slice of the
fire-control calibration harness, hashes everything they produce (the full
event stream, the ``sim_trace`` of traced runs, the harness' per-round rows)
and compares the hashes with the golden file next to this script. Reports CPU
time per case (``time.process_time`` - wall clock is meaningless on a loaded
machine).

An engine optimisation must leave every hash unchanged. A deliberate
behaviour change (new mechanic, rebalance) legitimately moves them: re-record
with ``--record`` on the commit that changes behaviour, never on a commit
that claims to be a pure optimisation.

    uv run python scripts/perf_check.py                # check all cases vs golden
    uv run python scripts/perf_check.py --record       # (re)write the golden file
    uv run python scripts/perf_check.py --cases s4_tac_trace,calib
    uv run python scripts/perf_check.py --profile s5_tac --top 40
                                                       # cProfile one case: call counts,
                                                       # tottime and cumulative tables
    uv run python scripts/perf_check.py --jobs 1       # serial (steadier CPU times)
"""

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_PATH = Path(__file__).with_name("perf_check_golden.json")

# name -> (kind, params). Different seeds draft different rosters; both
# rule-based doctrines; trace on (prediction shadow sim) and off.
CASES = {
    "s4_tac_trace": ("battle", dict(seed=4, doctrine="tactical", trace=True)),
    "s5_tac": ("battle", dict(seed=5, doctrine="tactical", trace=False)),
    "s5_tac_trace": ("battle", dict(seed=5, doctrine="tactical", trace=True)),
    "s7_leg_trace": ("battle", dict(seed=7, doctrine="legacy", trace=True)),
    "s11_leg": ("battle", dict(seed=11, doctrine="legacy", trace=False)),
    "s2_tac_300": ("battle", dict(seed=2, doctrine="tactical", trace=False, budget=300)),
    "calib": ("calib", dict(runs=3)),
}

# Functions whose call counts are printed by --profile (substring match on
# "file:function"); the tables below them show whatever is actually hot.
HOT_FUNCTIONS = (
    "simulation.py:step", "simulation.py:_update_projectiles",
    "simulation.py:_update_torpedoes", "simulation.py:_update_point_defense",
    "simulation.py:_update_ship", "simulation.py:get_enemy_ships",
    "physics.py:__add__", "physics.py:__sub__", "physics.py:__mul__",
    "physics.py:magnitude", "physics.py:normalized", "physics.py:dot",
    "physics.py:distance_to", "trajectory_prediction.py:predict",
)


def _sha(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def run_battle_case(seed: int, doctrine: str, trace: bool, budget: int = 200,
                    time_limit: float = 600.0, distance: float = 500.0) -> dict:
    """One free draft battle (heuristic captains, auto-draft, no admirals),
    built exactly like ``scripts/run_draft_battle.py --auto-draft
    --no-admirals --seed N --budget B --time-limit T --doctrine D [--trace]``."""
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    from src.llm.fleet_config import BattleFleetConfig
    from src.llm.fleet_draft import DEFAULT_MAX_SHIPS, auto_draft, draft_to_fleet_definition
    from src.llm.heuristic_captain import HEURISTIC_MODEL
    from src.llm.prompts import CaptainPersonality

    fleet_data = load_fleet_data()
    drafts = {
        faction: auto_draft(faction, budget=budget, max_ships=DEFAULT_MAX_SHIPS,
                            seed=seed + (0 if faction == "alpha" else 1))
        for faction in ("alpha", "beta")
    }
    with tempfile.TemporaryDirectory(prefix="perf_check_") as tmp:
        fleet_config = BattleFleetConfig(
            battle_name=f"Draft Battle ({budget} pts)",
            alpha_fleet=draft_to_fleet_definition(
                drafts["alpha"], distance, captain_model=HEURISTIC_MODEL, admiral_config=None),
            beta_fleet=draft_to_fleet_definition(
                drafts["beta"], distance, captain_model=HEURISTIC_MODEL, admiral_config=None),
            time_limit_s=time_limit,
            initial_distance_km=distance,
            record_battle=True,
            record_sim_trace=trace,
            personality_selection=False,
            use_notebooks=False,
        )
        battle_config = BattleConfig(
            initial_distance_km=distance,
            time_limit_s=time_limit,
            max_checkpoints=25,
            verbose=False,
            personality_selection=False,
            record_battle=True,
            record_sim_trace=trace,
            record_predictions=True,
            recording_dir=tmp,
            seed=seed,
            heuristic_doctrine={"alpha": doctrine, "beta": doctrine},
        )
        dummy = LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                                 personality=CaptainPersonality.BALANCED)
        runner = LLMBattleRunner(config=battle_config, alpha_config=dummy, beta_config=dummy,
                                 client=None, fleet_config=fleet_config)
        t0 = time.process_time()
        with contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_battle(fleet_data)
        cpu = time.process_time() - t0
        recording = json.loads(Path(result.recording_file).read_text())
        sim = runner.simulation

    # Full-precision end state: recorded events are rounded, and only the RNG
    # states prove that the number of draws did not change.
    end_state = {
        "t": repr(sim.current_time),
        "rng": _sha(sim.rng.getstate()),
        "damage_rng": _sha(sim._damage_rng.getstate()),
        "ships": {
            sid: [repr(v) for v in (
                ship.position.x, ship.position.y, ship.position.z,
                ship.velocity.x, ship.velocity.y, ship.velocity.z,
                ship.forward.x, ship.forward.y, ship.forward.z,
                ship.kinematic_state.mass_kg, ship.kinematic_state.propellant_kg,
                ship.heat_fraction, ship.hull_integrity,
                ship.is_destroyed, ship.is_dying)]
            for sid, ship in sim.ships.items()
        },
        "projectiles": [[repr(v) for v in (p.projectile.position.x, p.projectile.position.y,
                                           p.projectile.position.z, p.min_distance_to_target)]
                        for p in sim.projectiles],
        "torpedoes": [[repr(v) for v in (t.torpedo.position.x, t.torpedo.position.y,
                                         t.torpedo.position.z, t.torpedo.velocity.x,
                                         t.torpedo.velocity.y, t.torpedo.velocity.z)]
                      for t in sim.torpedoes],
    }
    events = recording["events"]
    hashes = {
        "events": _sha(events),
        "result": _sha([recording.get(k) for k in (
            "winner", "result_reason", "duration_s", "total_checkpoints",
            "alpha_ships_remaining", "beta_ships_remaining", "alpha_fleet", "beta_fleet")]),
        "end_state": _sha(end_state),
    }
    if trace:
        hashes["sim_trace"] = _sha(recording.get("sim_trace"))
    return {
        "hashes": hashes,
        "cpu_s": cpu,
        "info": f"{len(events)} events, {recording.get('duration_s'):.0f} s battle, "
                f"winner {recording.get('winner')}",
    }


def run_calib_case(runs: int) -> dict:
    """A slice of ``scripts/fire_control_calibration.py harness --runs N``
    (all target classes, one shooter): every resolved round, in order."""
    sys.path.insert(0, str(Path(__file__).parent))
    import fire_control_calibration as fcc

    jobs = fcc.harness_jobs(runs, fcc.HARNESS_CLASSES, (1,))
    jobs += fcc.harness_jobs(1, fcc.HARNESS_CLASSES, (3,))
    rows = []
    t0 = time.process_time()
    for job in jobs:
        rows.extend(fcc._harness_job(job))
    cpu = time.process_time() - t0
    return {
        "hashes": {"rows": _sha(rows)},
        "cpu_s": cpu,
        "info": f"{len(jobs)} engagements, {len(rows)} rounds",
    }


def run_case(name: str) -> dict:
    kind, params = CASES[name]
    return run_battle_case(**params) if kind == "battle" else run_calib_case(**params)


def profile_case(name: str, top: int) -> None:
    import cProfile
    import pstats

    # CPU clock, not the default wall clock: on a loaded machine wall-clock
    # profile seconds are inflated by however long the process was descheduled.
    prof = cProfile.Profile(time.process_time)
    prof.enable()
    out = run_case(name)
    prof.disable()
    print(f"{name}: {out['info']}; cpu {out['cpu_s']:.1f} s under cProfile "
          f"(table seconds are CPU seconds including profiler overhead)")
    stats = pstats.Stats(prof)
    print("\n-- call counts, hot functions --")
    rows = []
    for (fn, _line, func), (cc, nc, tt, ct, _callers) in stats.stats.items():
        key = f"{Path(fn).name}:{func}"
        if any(key == h for h in HOT_FUNCTIONS):
            rows.append((nc, tt, ct, key))
    for nc, tt, ct, key in sorted(rows, reverse=True):
        print(f"{nc:>12,d} calls  tottime {tt:7.2f}  cumtime {ct:7.2f}  {key}")
    for order in ("tottime", "cumulative"):
        print(f"\n-- top {top} by {order} --")
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).strip_dirs().sort_stats(order).print_stats(top)
        text = buf.getvalue()
        print(text[text.find("   ncalls"):].rstrip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=",".join(CASES), help="comma list (default: all)")
    ap.add_argument("--record", action="store_true", help="write the golden file")
    ap.add_argument("--jobs", type=int, default=4, help="parallel case processes (default 4)")
    ap.add_argument("--profile", metavar="CASE", help="cProfile one case and print the tables")
    ap.add_argument("--top", type=int, default=35)
    ap.add_argument("--run-case", metavar="CASE", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.run_case:
        print("PERF_CHECK_RESULT " + json.dumps(run_case(args.run_case)))
        return 0
    if args.profile:
        profile_case(args.profile, args.top)
        return 0

    names = [n for n in args.cases.split(",") if n]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        ap.error(f"unknown cases: {unknown}; known: {list(CASES)}")

    results = {}
    pending = list(names)
    running = {}
    while pending or running:
        while pending and len(running) < max(1, args.jobs):
            n = pending.pop(0)
            running[n] = subprocess.Popen(
                [sys.executable, __file__, "--run-case", n],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT))
        # Wait for the oldest running case (the others keep running meanwhile).
        n, proc = next(iter(running.items()))
        out, err = proc.communicate()
        del running[n]
        line = next((ln for ln in out.splitlines() if ln.startswith("PERF_CHECK_RESULT ")), None)
        if proc.returncode != 0 or line is None:
            print(f"{n}: FAILED to run\n{err[-3000:]}")
            results[n] = None
            continue
        results[n] = json.loads(line[len("PERF_CHECK_RESULT "):])

    golden = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}
    if args.record:
        for n, r in results.items():
            if r is not None:
                golden[n] = {"hashes": r["hashes"], "cpu_s": round(r["cpu_s"], 2),
                             "info": r["info"]}
        GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
        print(f"golden written: {GOLDEN_PATH}")

    ok = True
    total = total_ref = 0.0
    print(f"{'case':14s} {'cpu s':>8s} {'ref s':>8s} {'speedup':>8s}  hashes")
    for n in names:
        r = results[n]
        g = golden.get(n)
        if r is None:
            ok = False
            continue
        if g is None:
            verdict = "NO GOLDEN (run --record)"
            ok = False
            ref = None
        else:
            diff = [k for k in sorted(set(r["hashes"]) | set(g["hashes"]))
                    if r["hashes"].get(k) != g["hashes"].get(k)]
            verdict = "identical (" + ", ".join(sorted(r["hashes"])) + ")" if not diff \
                else "MISMATCH: " + ", ".join(diff)
            ok = ok and not diff
            ref = g.get("cpu_s")
        total += r["cpu_s"]
        total_ref += ref or 0.0
        speed = f"{ref / r['cpu_s']:7.2f}x" if ref else "       -"
        print(f"{n:14s} {r['cpu_s']:8.1f} {ref if ref is not None else float('nan'):8.1f} "
              f"{speed}  {verdict}   [{r['info']}]")
    if total_ref:
        print(f"{'TOTAL':14s} {total:8.1f} {total_ref:8.1f} {total_ref / total:7.2f}x")
    print("ALL HASHES IDENTICAL" if ok else "HASH CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
