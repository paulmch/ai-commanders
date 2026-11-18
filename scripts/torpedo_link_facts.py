#!/usr/bin/env python3
"""
Measured facts behind the torpedo launch-profile and datalink doctrine
(src/llm/torpedo_doctrine.py), written to tests/fixtures/torpedo_link_facts.json
and bound by tests/test_doctrine_accuracy.py::TestTorpedoLinkDoctrine.

- profile flights (scripts/torpedo_profile_sweep.py scenarios, PD off): from
  500 km at a battlecruiser holding course (time of flight, impact energy), at
  a frigate burning away at 3 g, and the reach of a reassignment 200 km out;
- seeded rule-based battles (scripts/torpedo_retarget_sweep.py, sprint with
  the datalink): share of rounds that lost their target in flight, share of
  those reassigned, share of the reassigned that hit.

    uv run python scripts/torpedo_link_facts.py --seeds 3 --workers 8 \
        --fixture tests/fixtures/torpedo_link_facts.json
"""
from __future__ import annotations

import os
import sys

_PINNED = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONHASHSEED": "0"}
if __name__ == "__main__" and any(os.environ.get(k) != v for k, v in _PINNED.items()):
    os.environ.update(_PINNED)
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torpedo_profile_sweep as P  # noqa: E402
import torpedo_retarget_sweep as S  # noqa: E402
from src import torpedo as T  # noqa: E402
from src import torpedo_link as L  # noqa: E402

PAIRS = [("swarm", "b3_beta"), ("swarm", "gunline"), ("torpedo", "b3_beta"),
         ("torpedo", "gunline"), ("b3_alpha", "b3_beta"), ("screen", "gunline")]


def profile_facts() -> dict:
    out = {}
    built = (T.RESERVE_CRUISE_CLOSING_KPS, T.RESERVE_TERMINAL_WINDOW_S)
    for name, setting in (("sprint", None), ("reserve", built)):
        f = P.flight(setting, 500.0)
        run = P.runner(setting, "frigate", 500.0)
        out[name] = {"tof_500_s": f["tof_s"], "gj_500": f["impact_gj"],
                     "runner_ff_gj": run["impact_gj"],
                     "reach_200_km": P.max_reach(setting, 200.0)}
    P._set(built)
    return out


def battle_facts(seeds: int, workers: int) -> dict:
    tasks = []
    for a, b in PAIRS:
        for seed in range(1, seeds + 1):
            for al, be in ((a, b), (b, a)):
                tasks.append(dict(key=f"facts|{al}|{be}|{seed}", option="p5_link", seed=seed,
                                  alpha=al, beta=be, alpha_roster=S.FLEETS[al],
                                  beta_roster=S.FLEETS[be], distance_km=500.0,
                                  time_limit_s=600.0))
    with mp.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        res = pool.map(S.run_battle, tasks)
    errs = [r for r in res if r.get("error")]
    if errs:
        raise SystemExit(f"{len(errs)} battles failed: {errs[0]['error']}")
    c = S.summarize(res)["p5_link"]
    return {"battles": c["battles"], "launched": c["launched"], "orphaned": c["orphaned"],
            "reassigned": c["retargeted"], "reassigned_hit": c["retargeted_hit_new_target"],
            "pct_orphaned": round(100 * c["orphaned"] / c["launched"]),
            "pct_reassigned": round(100 * c["retargeted"] / c["orphaned"]),
            "pct_reassigned_hit": round(100 * c["retargeted_hit_new_target"] / c["retargeted"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fixture", type=Path, default=None)
    args = ap.parse_args()
    facts = {
        "about": ("Measured facts behind src/llm/torpedo_doctrine.py (torpedo launch "
                  "profiles and the datalink), bound by tests/test_doctrine_accuracy.py. "
                  "Regenerate with the command below after a change to torpedo guidance, "
                  "the datalink or point defence."),
        "command": ("uv run python scripts/torpedo_link_facts.py --seeds 3 --workers 8 "
                    "--fixture tests/fixtures/torpedo_link_facts.json"),
        "constants": {"RESERVE_CRUISE_CLOSING_KPS": T.RESERVE_CRUISE_CLOSING_KPS,
                      "RESERVE_TERMINAL_WINDOW_S": T.RESERVE_TERMINAL_WINDOW_S,
                      "REASSIGN_RESERVE_BASE_KPS": L.REASSIGN_RESERVE_BASE_KPS,
                      "REASSIGN_RESERVE_PER_S_KPS": L.REASSIGN_RESERVE_PER_S_KPS},
        "profiles": profile_facts(),
        "battles": battle_facts(args.seeds, args.workers),
    }
    print(json.dumps(facts, indent=1))
    if args.fixture:
        args.fixture.write_text(json.dumps(facts, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
