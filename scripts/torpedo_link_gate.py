#!/usr/bin/env python3
"""
Re-derive the datalink's reassignment reserve from flown rounds.

A cruising round (12 km/s, a given delta-v left) is reassigned exactly as the
datalink does it (CombatSimulation._reassign_torpedo: fresh engagement,
cruise floor released) onto a hull at a given range / offset / speed, PD off,
and flown to the end. Per flight: the engine's can_intercept answer at the
moment of reassignment (time to go, fuel left after the steer) and whether
the round hit. The target either holds course (no orders) or flies EVASIVE
(the rule layer's jink) or burns flat out across the line of sight
(HEADING, throttle 1) - the worst case the reserve must pay for.

The table printed is hit rate by (time-to-go bucket, fuel-left-after-steer
bucket) for each target behaviour; src/torpedo_link.py's
reassign_reserve_kps(t) = BASE + PER_S * t should sit where the evading
hit rate is acceptable.

    uv run python scripts/torpedo_link_gate.py --workers 6 --json out.json
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
import math
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.battle_runner import load_fleet_data  # noqa: E402
from src.physics import Vector3D  # noqa: E402
from src.simulation import (  # noqa: E402
    CombatSimulation, Maneuver, ManeuverType, SimulationEventType, TorpedoInFlight,
    _build_torpedo_specs, create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo  # noqa: E402

FLEET = load_fleet_data()
SPECS = _build_torpedo_specs(FLEET["weapon_types"]["torpedo_launcher"])


def fly(case):
    dv, x_km, y_km, vx, typ, evade, seed = case
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)

    def mk(sid, t, fac, pos, vel=Vector3D(0, 0, 0)):
        s = create_ship_from_fleet_data(sid, t, fac, FLEET, position=pos, velocity=vel,
                                        forward=Vector3D(-1, 0, 0) if fac == "beta"
                                        else Vector3D(1, 0, 0))
        s.point_defense = []
        sim.add_ship(s)
        return s
    mk("a", "corvette", "alpha", Vector3D(-3_000e3, 0, 0))
    cand = mk("c", typ, "beta", Vector3D(x_km * 1e3, y_km * 1e3, 0), Vector3D(vx, 0, 0))
    if evade == "evasive":
        sim.inject_command("c", Maneuver(ManeuverType.EVASIVE, start_time=0.0, throttle=1.0))
    elif evade == "burn":
        # flat out across the line of sight (after its turn), away from the
        # side the round comes in on
        side = -1.0 if y_km > 0 else 1.0
        sim.inject_command("c", Maneuver(ManeuverType.HEADING, start_time=0.0, throttle=1.0,
                                         heading_direction={"x": 0.0, "y": -side, "z": 0.0}))
    torp = Torpedo(specs=SPECS, position=Vector3D(0, 0, 0), velocity=Vector3D(12_000, 0, 0),
                   target_id="c", guidance_mode=GuidanceMode.COLLISION)
    torp.armed = True
    torp.current_mass_kg = SPECS.dry_mass_kg * math.exp(dv / SPECS.exhaust_velocity_kps)
    torp.remaining_delta_v_kps = dv
    ok, t, left = torp.can_intercept(cand.position, cand.velocity)
    if not ok or t > 600:
        return None
    tf = TorpedoInFlight(torpedo_id="t", torpedo=torp, source_ship_id="a")
    sim.torpedoes.append(tf)
    sim._reassign_torpedo(tf, cand, "target_lost", "gate")
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.torpedoes and sim.current_time < t + 120:
            sim.step()
    hit = any(e.event_type == SimulationEventType.TORPEDO_IMPACT for e in sim.events)
    return {"dv": dv, "x": x_km, "y": y_km, "vx": vx, "type": typ, "evade": evade,
            "seed": seed, "t": t, "left": left, "hit": hit}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    cases = list(itertools.product(
        (0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0),
        (60, 120, 200, 300, 450),
        (0, 8, 20, 40),
        (0.0, 1500.0, -1500.0),
        ("battlecruiser", "frigate"),
        ("hold", "evasive", "burn"),
        (1,)))
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = [r for r in pool.map(fly, cases, chunksize=8) if r]
    if args.json:
        args.json.write_text(json.dumps(res))
    tb = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 80)]
    lb = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 9)]
    for evade in ("hold", "evasive", "burn"):
        for typ in ("battlecruiser", "frigate"):
            print(f"\n{typ} {evade}: hit rate (n) by "
                  f"time to go (rows, s) x fuel left after the steer (cols, km/s)")
            print("        " + "".join(f"{f'{a:g}-{b:g}':>13s}" for a, b in lb))
            for t0, t1 in tb:
                row = f"{t0:>3d}-{t1:<3d} "
                for l0, l1 in lb:
                    sel = [r for r in res if r["evade"] == evade and r["type"] == typ
                           and t0 <= r["t"] < t1 and l0 <= r["left"] < l1]
                    row += (f"{sum(r['hit'] for r in sel) / len(sel):>8.0%} ({len(sel):>2d})"
                            if sel else f"{'-':>13s}")
                print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
