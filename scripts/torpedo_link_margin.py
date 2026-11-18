#!/usr/bin/env python3
"""
Derive torpedo_link.LINK_KILL_MARGIN_ROUNDS from flown strikes.

k rounds launched together from a torpedo cruiser at ONE hull with its point
defence on (weapon arcs on, no orders: the hull flies its rule layer), for
k = 1 .. 14, sprint and reserve, a launcher at rest or closing at 3 km/s,
seeds 1..4. Per cell: need = torpedo_link.need_for(...).total read off the
board on the first tick (the estimate the datalink and the commanders use),
and the kill rate at each k. The margin m printed is the smallest m with
P(kill | need + m) >= P(kill | 14) in every cell - surplus diversion that
keeps need + m rounds on a target never costs a kill the full strike had.

    uv run python scripts/torpedo_link_margin.py --workers 8 [--json out.json]
"""
from __future__ import annotations

import os
import sys

_PINNED = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
if __name__ == "__main__" and any(os.environ.get(k) != v for k, v in _PINNED.items()):
    os.environ.update(_PINNED)
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import contextlib
import io
import itertools
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import torpedo_link as L  # noqa: E402
from src.llm.battle_runner import load_fleet_data  # noqa: E402
from src.physics import Vector3D  # noqa: E402
from src.simulation import CombatSimulation, SimulationEventType as ET, create_ship_from_fleet_data  # noqa: E402

FLEET = load_fleet_data()
CLASSES = ("frigate", "destroyer", "cruiser", "battlecruiser", "battleship")
KMAX = 14


def strike(case):
    cls, profile, vc, k, seed = case
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    sim.echo = False
    sh = create_ship_from_fleet_data("a", "cruiser_torpedo", "alpha", FLEET,
                                     position=Vector3D(0, 0, 0), velocity=Vector3D(vc, 0, 0),
                                     forward=Vector3D(1, 0, 0))
    sh.point_defense = []
    sim.add_ship(sh)
    tg = create_ship_from_fleet_data("b", cls, "beta", FLEET, position=Vector3D(500e3, 0, 0),
                                     forward=Vector3D(-1, 0, 0))
    sim.add_ship(tg)
    for _ in range(k):
        for l in sh.torpedo_launchers:
            l.last_launch_time = -1e9
            l.current_magazine = max(l.current_magazine, 1)
        cmd = {"target_id": "b"}
        if profile != "sprint":
            cmd["profile"] = profile
        sim._handle_torpedo_command(sh, cmd)
    need = None
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.torpedoes and sim.current_time < 600:
            sim.step()
            if need is None:
                board = sim.torpedo_board("alpha")
                if "b" in board:
                    need = board["b"]["need"].total
    killed = any(e.event_type in (ET.SHIP_DYING, ET.SHIP_DESTROYED) and e.ship_id == "b"
                 for e in sim.events)
    hits = sum(1 for e in sim.events if e.event_type == ET.TORPEDO_IMPACT)
    return {"cls": cls, "profile": profile, "vc": vc, "k": k, "seed": seed,
            "need": need, "killed": killed, "hits": hits}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    cases = list(itertools.product(CLASSES, ("sprint", "reserve"), (0.0, 3000.0),
                                   range(1, KMAX + 1), (1, 2, 3, 4)))
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = pool.map(strike, cases, chunksize=4)
    if args.json:
        args.json.write_text(json.dumps(res))
    cells = defaultdict(lambda: defaultdict(list))
    needs = defaultdict(list)
    for r in res:
        key = (r["cls"], r["profile"], r["vc"])
        cells[key][r["k"]].append(r["killed"])
        if r["need"] is not None:
            needs[key].append(r["need"])
    worst = 0
    print("class profile closing: need (board) | kill rate by k | k needed for the full-strike rate")
    for key in sorted(cells):
        rate = {k: sum(v) / len(v) for k, v in cells[key].items()}
        need = max(set(needs[key]), key=needs[key].count) if needs[key] else None
        full = rate[KMAX]
        m = next((m for m in range(0, KMAX) if need is not None and need + m <= KMAX
                  and all(rate[k] >= full for k in range(need + m, KMAX + 1))), None)
        worst = max(worst, m if m is not None else KMAX)
        print(f"{key[0]:13s} {key[1]:7s} {key[2]:5.0f}: need {need} | "
              + " ".join(f"{k}:{rate[k]:.2f}" for k in sorted(rate)) + f" | margin {m}")
    print(f"\nLINK_KILL_MARGIN_ROUNDS (worst cell) = {worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
