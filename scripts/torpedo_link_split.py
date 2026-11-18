#!/usr/bin/env python3
"""
Does surplus diversion cost kills on the ORIGINAL target? (review defect 1,
after the reviewer's split.py.) A torpedo cruiser fires n rounds at once at
hull b1 500 km out; an identical hull b2 sits 20 / 40 / 80 km abeam of it.
Launcher at rest, closing 3 km/s or opening 1 km/s; PD on both targets.
Three modes: no datalink, the link without surplus diversion, the full link.
Per cell: b1 kill rate, kills of both, hits.

    uv run python scripts/torpedo_link_split.py --workers 12 [--json out.json]
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
MODES = ("nolink", "link_nosurplus", "link")


def run(case):
    hull, n, prof, mode, off, vc, seed = case
    L.LINK_SURPLUS_DIVERSION = mode == "link"
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    sim.echo = False
    sim.torpedo_datalink = mode != "nolink"
    sh = create_ship_from_fleet_data("a1", "cruiser_torpedo", "alpha", FLEET,
                                     position=Vector3D(0, 0, 0), velocity=Vector3D(vc, 0, 0),
                                     forward=Vector3D(1, 0, 0))
    sh.point_defense = []
    sim.add_ship(sh)
    ts = []
    for i, y in enumerate((0.0, off * 1e3)):
        t = create_ship_from_fleet_data(f"b{i + 1}", hull, "beta", FLEET,
                                        position=Vector3D(500e3, y, 0),
                                        forward=Vector3D(-1, 0, 0))
        sim.add_ship(t)
        ts.append(t)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            for l in sh.torpedo_launchers:
                l.last_launch_time = -1e9
                l.current_magazine = max(l.current_magazine, 1)
            cmd = {"target_id": "b1"}
            if prof != "sprint":
                cmd["profile"] = prof
            sim._handle_torpedo_command(sh, cmd)
        for _ in range(400):
            sim.step()
            if not sim.torpedoes:
                break
    div = sum(1 for e in sim.events if e.event_type == ET.TORPEDO_RETARGETED
              and e.data.get("reason") == "datalink_surplus")
    return {"case": case, "div": div, "b1": bool(ts[0].is_destroyed or ts[0].is_dying),
            "kills": sum(1 for s in ts if s.is_destroyed or s.is_dying),
            "hits": sum(1 for e in sim.events if e.event_type == ET.TORPEDO_IMPACT)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    cases = list(itertools.product(("destroyer", "cruiser", "battlecruiser", "battleship"),
                                   (4, 6, 8, 10, 12), ("reserve",), MODES, (20, 40, 80),
                                   (0.0, 3000.0, -1000.0), (1, 2, 3)))
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=50) as pool:
        res = pool.map(run, cases, chunksize=4)
    if args.json:
        args.json.write_text(json.dumps(res))
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in res:
        hull, n, prof, mode, off, vc, seed = r["case"]
        a = agg[(hull, n, vc, mode)]
        a[0] += 1
        a[1] += r["b1"]
        a[2] += r["kills"]
        a[3] += r["hits"]
        a[4] += r["div"]
    print("hull n launcher: b1 kill rate / kills per run / hits / diverted, per mode "
          + " | ".join(MODES))
    worse = []
    for hull in ("destroyer", "cruiser", "battlecruiser", "battleship"):
        for n in (4, 6, 8, 10, 12):
            for vc in (0.0, 3000.0, -1000.0):
                cells = [agg[(hull, n, vc, m)] for m in MODES]
                txt = " | ".join(f"{c[1] / c[0]:.2f} {c[2] / c[0]:.2f} {c[3] / c[0]:.1f} {c[4] / c[0]:.1f}"
                                 for c in cells)
                print(f"{hull:13s} {n:2d} {vc:6.0f}: {txt}")
                if cells[2][1] < cells[1][1]:
                    worse.append((hull, n, vc, cells[1][1] / cells[1][0], cells[2][1] / cells[2][0]))
    tot = {m: [sum(agg[k][i] for k in agg if k[3] == m) for i in range(5)] for m in MODES}
    print("totals (runs, b1 kills, kills, hits, diverted):", tot)
    print("cells where the full link killed b1 LESS often than the link without diversion:", worse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
