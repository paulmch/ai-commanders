#!/usr/bin/env python3
"""
The evade overlay on a burning order, measured: what a ship ordered to burn
actually flies under gun fire, from engine truth per tick (applied throttle,
nose angle to the ordered burn line, displacement along it, hits by facing).
The numbers the commander doctrine quotes about the overlay come from here
(tests/test_doctrine_accuracy.py re-runs a slice).

Scenarios (seeded, no LLM; every gun FIRE_IMMEDIATE at the target):
  cross  : target HEADING @100% across the line of sight (nose already on the
           heading), a battlecruiser + 2 destroyers PADLOCKed 120-200 km away, 150 s
  escape : target HEADING @100% straight away, a battlecruiser + a destroyer
           INTERCEPTing from 150 km, 240 s
  merge  : target closing at 2 km/s on a battlecruiser + a destroyer 200 km away
           (PADLOCK), HEADING @100% against the closure (cancel it), 200 s
  escape1, merge1: the same against one destroyer (a fight the target can survive,
           so its closest approach is not cut short by its death)
Each with the overlay off, 'dodge' and 'full'.

    OMP_NUM_THREADS=1 uv run python scripts/evade_burn_measure.py --out /tmp/eb.json
    uv run python scripts/evade_burn_measure.py --scenarios cross --hulls destroyer --seeds 1-12

Columns: on_order = share of the ticks under fire (a gun round inbound) with
the drive at >= half the order's throttle and the nose within 30 deg of the
burn line; along = mean of applied throttle x cos(nose to line) over those
ticks; d60/d120/d240 = km moved along the line; close60/close120 = closing
speed (km/s) on the nearest shooter at 60 / 120 s (rows alive then); min_range /
range_end = km to
the nearest shooter; hit_pct = hits / rounds fired at the target; pen =
penetrating hits; loc = hits by facing; dead = runs the target died in.
"""
import argparse
import contextlib
import io
import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.firecontrol import WeaponsCommand, WeaponsOrder  # noqa: E402
from src.physics import Vector3D  # noqa: E402
from src.simulation import (CombatSimulation, Maneuver, ManeuverType,  # noqa: E402
                            SimulationEventType, create_ship_from_fleet_data)

FLEET = json.loads((ROOT / "data" / "fleet_ships.json").read_text())
SEEDS = list(range(1, 13))
EVADES = (None, "dodge", "full")
CROSS_HULLS = ("corvette", "destroyer", "battlecruiser", "dreadnought")
SCENARIOS = ("cross", "escape", "merge", "escape1", "merge1")
RUN_HULLS = ("destroyer", "battlecruiser")


def fire_orders(ship, target_id):
    ship.weapons_orders = {
        slot: WeaponsOrder(WeaponsCommand.FIRE_IMMEDIATE, slot, target_id, min_hit_probability=0.0)
        for slot in ship.weapons if not slot.startswith("pd_")}


def run(job):
    scen, hull, evade, seed = job
    rng = random.Random(f"{scen}-{hull}-{seed}")
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=seed)
    if scen == "cross":
        d = Vector3D(0, 1, 0)
        rng_km = rng.uniform(120, 200)
        tgt = create_ship_from_fleet_data("target", hull, "alpha", FLEET, position=Vector3D(0, 0, 0),
                                          velocity=Vector3D(0, 0, 0), forward=d)
        centre = Vector3D(rng_km * 1000.0, 0, 0)
        dur = 150.0
        shooter_mode = "PADLOCK"
    elif scen in ("escape", "escape1"):
        d = Vector3D(-1, 0, 0)
        tgt = create_ship_from_fleet_data("target", hull, "alpha", FLEET, position=Vector3D(0, 0, 0),
                                          velocity=Vector3D(0, 0, 0), forward=d)
        centre = Vector3D(150e3, 0, 0)
        dur = 240.0
        shooter_mode = "INTERCEPT"
    else:  # merge, merge1
        d = Vector3D(-1, 0, 0)
        tgt = create_ship_from_fleet_data("target", hull, "alpha", FLEET, position=Vector3D(0, 0, 0),
                                          velocity=Vector3D(2000.0, 0, 0), forward=d)
        centre = Vector3D(200e3, 0, 0)
        dur = 200.0
        shooter_mode = "PADLOCK"
    sim.add_ship(tgt)
    shooters = []
    roster = {"cross": ("battlecruiser", "destroyer", "destroyer"), "escape1": ("destroyer",),
              "merge1": ("destroyer",)}.get(scen, ("battlecruiser", "destroyer"))
    for i, st in enumerate(roster):
        ang = math.radians((i - 1) * rng.uniform(5, 20))
        off = Vector3D(math.cos(ang), math.sin(ang), rng.uniform(-0.05, 0.05)) * centre.magnitude
        pos = off
        s = create_ship_from_fleet_data(f"shooter{i}", st, "beta", FLEET, position=pos,
                                        velocity=Vector3D(0, 0, 0),
                                        forward=(tgt.position - pos).normalized())
        sim.add_ship(s)
        mt = ManeuverType.INTERCEPT if shooter_mode == "INTERCEPT" else ManeuverType.PADLOCK
        s.current_maneuver = Maneuver(mt, 0, duration=1e9, throttle=1.0 if mt == ManeuverType.INTERCEPT else 0.0,
                                      target_id="target")
        fire_orders(s, "target")
        shooters.append(s)
    tgt.current_maneuver = Maneuver(ManeuverType.HEADING, 0, duration=1e9, throttle=1.0,
                                    heading_direction={"x": d.x, "y": d.y, "z": d.z}, evade=evade)
    p0 = tgt.position
    a_full = tgt.kinematic_state.max_acceleration_ms2()
    under = burn_half = burn_full = on_order = 0
    along = 0.0
    disp = {}
    closing = {}
    min_range = float("inf")
    hits = []
    shots = 0
    died = None
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.current_time < dur and not (tgt.is_destroyed or tgt.is_dying):
            inbound = any(p.target_ship_id == "target" for p in sim.projectiles)
            evs = sim.step()
            for ev in evs:
                if ev.event_type == SimulationEventType.PROJECTILE_LAUNCHED and ev.target_id == "target":
                    shots += 1
                elif ev.event_type == SimulationEventType.PROJECTILE_IMPACT and ev.target_id == "target":
                    dd = ev.data or {}
                    hits.append((dd.get("hit_location"), bool(dd.get("penetrated")),
                                 float(dd.get("damage_to_hull_gj") or 0.0)))
            if inbound and not (tgt.is_destroyed or tgt.is_dying):
                under += 1
                app = tgt.applied_throttle or 0.0
                burn_half += app >= 0.5
                burn_full += app >= 0.99
                along += app * tgt.forward.dot(d)
                on_order += app >= 0.5 and tgt.forward.dot(d) >= math.cos(math.radians(30))
            t = round(sim.current_time)
            if t in (60, 120, 240):
                disp[t] = (tgt.position - p0).dot(d) / 1000.0
            if t in (60, 120):
                near = min(shooters, key=lambda s: (tgt.position - s.position).magnitude)
                rel = near.position - tgt.position
                closing[t] = (tgt.velocity - near.velocity).dot(rel.normalized()) / 1000.0
            r = min((tgt.position - s.position).magnitude for s in shooters) / 1000.0
            min_range = min(min_range, r)
        if tgt.is_destroyed or tgt.is_dying:
            died = sim.current_time
    rng_end = min((tgt.position - s.position).magnitude for s in shooters) / 1000.0
    return dict(scen=scen, hull=hull, evade=evade, seed=seed, under=under, burn_half=burn_half,
                burn_full=burn_full, on_order=on_order, closing=closing, along=along, disp=disp, min_range=min_range, range_end=rng_end,
                hits=hits, shots=shots, died=died, a=a_full)


def jobs(scens, hulls=None, seeds=SEEDS, evades=EVADES):
    out = []
    for scen in scens:
        for hull in hulls or (CROSS_HULLS if scen == "cross" else RUN_HULLS):
            for evade in evades:
                for seed in seeds:
                    out.append((scen, hull, evade, seed))
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["scen"], r["hull"], r["evade"]), []).append(r)
    lines = []
    for (scen, hull, evade), g in groups.items():
        under = sum(r["under"] for r in g)
        hits = [h for r in g for h in r["hits"]]
        pen = sum(1 for h in hits if h[1])
        gj = sum(h[2] for h in hits)
        loc = {}
        for h in hits:
            loc[h[0]] = loc.get(h[0], 0) + 1
        shots = sum(r["shots"] for r in g)
        dead = sum(1 for r in g if r["died"] is not None)

        def mean_disp(t):
            v = [r["disp"].get(t, r["disp"].get(str(t))) for r in g]
            v = [x for x in v if x is not None]
            return (sum(v) / len(v)) if v else float("nan")
        line = dict(scen=scen, hull=hull, evade=str(evade), runs=len(g), under_s=under,
                    burn_half=round(100 * sum(r["burn_half"] for r in g) / max(1, under), 1),
                    burn_full=round(100 * sum(r["burn_full"] for r in g) / max(1, under), 1),
                    on_order=round(100 * sum(r["on_order"] for r in g) / max(1, under), 1),
                    along=round(100 * sum(r["along"] for r in g) / max(1, under), 1),
                    d60=round(mean_disp(60), 1), d120=round(mean_disp(120), 1),
                    close60=_mean([r["closing"].get(60, r["closing"].get("60")) for r in g]),
                    close120=_mean([r["closing"].get(120, r["closing"].get("120")) for r in g]),
                    d240=round(mean_disp(240), 1),
                    min_range=round(sum(r["min_range"] for r in g) / len(g), 1),
                    min_range_lo=round(min(r["min_range"] for r in g), 1),
                    range_end=round(sum(r["range_end"] for r in g) / len(g), 1),
                    shots=shots, hits=len(hits), hit_pct=round(100 * len(hits) / max(1, shots), 2),
                    pen=pen, hull_gj=round(gj, 1), loc=loc, dead=dead)
        lines.append(line)
    return lines


def _seeds(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--hulls", default=None, help="comma list (default: per scenario)")
    ap.add_argument("--seeds", default="1-12")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None, help="write rows + summary JSON here")
    args = ap.parse_args()
    todo = jobs(args.scenarios.split(","), args.hulls.split(",") if args.hulls else None,
                _seeds(args.seeds))
    with Pool(args.jobs) as pool:
        rows = pool.map(run, todo, chunksize=2)
    summ = summarise(rows)
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "summary": summ}, indent=0))
    for s in summ:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
