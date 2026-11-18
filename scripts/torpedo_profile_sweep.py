#!/usr/bin/env python3
"""
Torpedo launch-profile sweep: what the `reserve` profile costs and buys
against `sprint`, for candidate settings of its two constants
(src/torpedo.py RESERVE_CRUISE_CLOSING_KPS, RESERVE_TERMINAL_WINDOW_S).

Seeded engine scenarios (corvette launcher, battlecruiser targets, no orders
on any hull: every ship flies its automatic rule-based layer only):

  flight     one round from 300 / 500 / 800 km at a coasting target, PD off:
             time of flight, impact speed, impact energy, delta-v left at
             launch+60 s (what a lost target would leave it to steer with)
  pd         a salvo of N rounds at a battlecruiser with its own PD (weapon
             arcs on), 500 km: rounds blinded, rounds through, per turret
  reach      a round flying at a target 100 / 200 / 300 km ahead; the target
             dies, a second hull sits abeam of it at offset y; the round is
             reassigned exactly as the datalink does (CombatSimulation.
             _reassign_torpedo, floor released), PD off. Largest y that
             still hits (bisection on the flown outcome)
  evade      one round from 500 km at a target on EVASIVE: hit?
  wall_pd    a salvo at a battleship behind a three-frigate PD screen
  wall_salvo 15 rounds (5 corvettes x 3) on the middle hull of a
             three-battlecruiser wall, datalink off / on

    uv run python scripts/torpedo_profile_sweep.py --settings 6:12,8:12,10:12,8:8,8:20
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
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import torpedo as T  # noqa: E402
from src.llm.battle_runner import load_fleet_data  # noqa: E402
from src.physics import Vector3D  # noqa: E402
from src.simulation import (  # noqa: E402
    CombatSimulation, Maneuver, ManeuverType, SimulationEventType,
    create_ship_from_fleet_data,
)

FLEET = load_fleet_data()


def _set(setting: Optional[Tuple[float, float]]) -> str:
    if setting is None:
        return "sprint"
    T.RESERVE_CRUISE_CLOSING_KPS, T.RESERVE_TERMINAL_WINDOW_S = setting
    return "reserve"


def _ship(sim, sid, typ, fac, pos, vel=None, fwd=None, pd=True):
    s = create_ship_from_fleet_data(sid, typ, fac, FLEET, position=pos,
                                    velocity=vel or Vector3D(0, 0, 0),
                                    forward=fwd or (Vector3D(-1, 0, 0) if fac == "beta"
                                                    else Vector3D(1, 0, 0)))
    if not pd:
        s.point_defense = []
    sim.add_ship(s)
    return s


def _run(sim, horizon):
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.torpedoes and sim.current_time < horizon:
            sim.step()


def _run_steps(sim, n):
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            if sim.torpedoes:
                sim.step()


def _launch(sim, shooter, target, profile, n=1):
    launched = 0
    for _ in range(n):
        for launcher in shooter.torpedo_launchers:
            launcher.last_launch_time = -1e9
        if sim._handle_torpedo_command(shooter, {"target_id": target, "profile": profile}):
            launched += 1
    return launched


def flight(setting, range_km: float, seed: int = 1, evade: bool = False,
           launcher_kps: float = 0.0) -> Dict[str, Any]:
    """``launcher_kps``: the launcher's speed toward the target (negative:
    opening) - a round inherits it."""
    profile = _set(setting)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    a = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0),
              vel=Vector3D(launcher_kps * 1000.0, 0, 0), pd=False)
    b = _ship(sim, "b", "battlecruiser", "beta", Vector3D(range_km * 1e3, 0, 0), pd=False)
    if evade:
        sim.inject_command("b", Maneuver(ManeuverType.EVASIVE, start_time=0.0, throttle=1.0))
    _launch(sim, a, "b", profile)
    tf = sim.torpedoes[0]
    dv60 = None
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.torpedoes and sim.current_time < 900:
            sim.step()
            if dv60 is None and sim.current_time >= 60 and sim.torpedoes:
                dv60 = tf.torpedo.remaining_delta_v_kps
    hit = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_IMPACT]
    return {"tof_s": round(hit[0].timestamp, 1) if hit else None,
            "impact_kps": round(hit[0].data["impact_speed_kps"], 2) if hit else None,
            "impact_gj": round(hit[0].data["total_damage_gj"], 1) if hit else None,
            "dv_at_60s": None if dv60 is None else round(dv60, 2)}


def pd_salvo(setting, n: int, seed: int = 1, range_km: float = 500.0) -> Dict[str, Any]:
    profile = _set(setting)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    a = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0), pd=False)
    b = _ship(sim, "b", "battlecruiser", "beta", Vector3D(range_km * 1e3, 0, 0))
    turrets = len(b.point_defense)
    _launch(sim, a, "b", profile, n)
    _run(sim, 900)
    ev = sim.events
    blinded = len({e.data.get("torpedo_id") for e in ev
                   if e.event_type == SimulationEventType.PD_TORPEDO_DISABLED})
    hits = len([e for e in ev if e.event_type == SimulationEventType.TORPEDO_IMPACT])
    return {"n": n, "turrets": turrets, "blinded": blinded, "hits": hits}


def runner(setting, typ: str, range_km: float = 500.0, seed: int = 1) -> Dict[str, Any]:
    """One round at a hull that burns straight away from the launcher at full
    thrust from the moment of launch, PD off: does it connect, when, how hard."""
    profile = _set(setting)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    a = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0), pd=False)
    _ship(sim, "b", typ, "beta", Vector3D(range_km * 1e3, 0, 0), fwd=Vector3D(1, 0, 0), pd=False)
    sim.inject_command("b", Maneuver(ManeuverType.HEADING, start_time=0.0, throttle=1.0,
                                     heading_direction={"x": 1.0, "y": 0.0, "z": 0.0}))
    _launch(sim, a, "b", profile)
    _run(sim, 900)
    hit = [e for e in sim.events if e.event_type == SimulationEventType.TORPEDO_IMPACT]
    return {"tof_s": round(hit[0].timestamp, 1) if hit else None,
            "impact_gj": round(hit[0].data["total_damage_gj"], 1) if hit else None}


def wall_pd(setting, n: int, seed: int = 1, range_km: float = 500.0) -> Dict[str, Any]:
    """A salvo of n at a battleship screened by three frigates 15 km ahead
    of it (3 + 6 PD turrets bear), weapon arcs on."""
    profile = _set(setting)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    a = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0), pd=False)
    x = range_km * 1e3
    _ship(sim, "b", "battleship", "beta", Vector3D(x, 0, 0))
    for i, y in enumerate((-12e3, 0.0, 12e3)):
        _ship(sim, f"f{i}", "frigate", "beta", Vector3D(x - 15e3, y, 3e3))
    _launch(sim, a, "b", profile, n)
    _run(sim, 900)
    ev = sim.events
    blinded = len({e.data.get("torpedo_id") for e in ev
                   if e.event_type == SimulationEventType.PD_TORPEDO_DISABLED})
    hits = len([e for e in ev if e.event_type == SimulationEventType.TORPEDO_IMPACT])
    return {"n": n, "blinded": blinded, "hits": hits}


def wall_salvo(setting, link: bool, seed: int = 1) -> Dict[str, Any]:
    """The p5w5_b3 pattern: five corvettes put 3 rounds each (15) on the
    middle hull of a three-battlecruiser wall 500 km out (PD on, no orders).
    Rounds that land, hulls killed, rounds that hit a dying hulk, rounds
    that ended without a hit."""
    profile = _set(setting)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    sim.torpedo_datalink = link
    shooters = [_ship(sim, f"a{i}", "corvette", "alpha", Vector3D(0, (i - 2) * 12e3, 0), pd=False)
                for i in range(5)]
    for i, y in enumerate((-12e3, 0.0, 12e3)):
        _ship(sim, f"b{i}", "battlecruiser", "beta", Vector3D(500e3, y, 0))
    for s in shooters:
        _launch(sim, s, "b1", profile, 3)
    hulk_hits = 0
    hit_resolve = sim._resolve_torpedo_hit

    def spy(tf, target):
        nonlocal hulk_hits
        hulk_hits += bool(target.is_dying)
        return hit_resolve(tf, target)
    sim._resolve_torpedo_hit = spy
    _run(sim, 600)
    ev = sim.events
    hits = len([e for e in ev if e.event_type == SimulationEventType.TORPEDO_IMPACT])
    kills = len({e.ship_id for e in ev if e.event_type in (
        SimulationEventType.SHIP_DESTROYED, SimulationEventType.SHIP_DYING)})
    rt = [e for e in ev if e.event_type == SimulationEventType.TORPEDO_RETARGETED]
    return {"hits": hits, "kills": kills, "hulk_hits": hulk_hits, "fired": 15,
            "retargeted": len({e.data["torpedo_id"] for e in rt}),
            "surplus": len({e.data["torpedo_id"] for e in rt
                            if e.data.get("reason") == "datalink_surplus"}),
            "t_end": sim.current_time}


def reach(setting, to_go_km: float, offset_km: float, seed: int = 1,
          launcher_kps: float = 0.0) -> bool:
    """Round cruising at its target; the target dies with to_go_km left; a
    second hull abeam at offset_km gets the round by datalink reassignment."""
    profile = _set(setting)
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
    a = _ship(sim, "a", "corvette", "alpha", Vector3D(0, 0, 0),
              vel=Vector3D(launcher_kps * 1000.0, 0, 0), pd=False)
    far = 800e3
    b1 = _ship(sim, "b1", "battlecruiser", "beta", Vector3D(far, 0, 0), pd=False)
    _launch(sim, a, "b1", profile)
    tf = sim.torpedoes[0]
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.torpedoes and tf.torpedo.position.distance_to(b1.position) > to_go_km * 1e3:
            sim.step()
    if not sim.torpedoes:
        return False
    b2 = _ship(sim, "b2", "battlecruiser", "beta",
               Vector3D(b1.position.x, offset_km * 1e3, 0), pd=False)
    b1.is_destroyed = True
    # the datalink's own decision (gate included): the advertised reach is the
    # flown reach of rounds the gate accepts
    _run_steps(sim, 1)
    if not any(e.event_type == SimulationEventType.TORPEDO_RETARGETED for e in sim.events):
        return False
    _run(sim, sim.current_time + 300)
    return any(e.event_type == SimulationEventType.TORPEDO_IMPACT and e.target_id == "b2"
               for e in sim.events)


def max_reach(setting, to_go_km: float, launcher_kps: float = 0.0) -> float:
    lo, hi = 0.0, 400.0
    if not reach(setting, to_go_km, 0.5, launcher_kps=launcher_kps):
        return 0.0
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        if reach(setting, to_go_km, mid, launcher_kps=launcher_kps):
            lo = mid
        else:
            hi = mid
    return round(lo, 1)


def one_setting(setting) -> Dict[str, Any]:
    out: Dict[str, Any] = {"setting": setting}
    for r in (300, 500, 800):
        out[f"flight_{r}"] = flight(setting, r)
    out["evade_500"] = [bool(flight(setting, 500, seed=s, evade=True)["tof_s"]) for s in (1, 2, 3)]
    out["pd"] = {n: [pd_salvo(setting, n, seed=s) for s in (1, 2)] for n in (1, 2, 3, 4, 6)}
    out["reach"] = {k: max_reach(setting, k) for k in (100, 200, 300)}
    out["runner"] = {f"{t}_{r}": runner(setting, t, r) for t in ("battlecruiser", "frigate")
                     for r in (300, 500)}
    out["wall_pd"] = {n: [wall_pd(setting, n, seed=s) for s in (1, 2)] for n in (3, 6, 9)}
    out["wall_salvo"] = {("link" if link else "nolink"): [wall_salvo(setting, link, seed=s)
                                                         for s in (1, 2, 3)]
                         for link in (False, True)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", default="6:12,8:12,10:12,8:6,8:20",
                    help="floor_kps:window_s,... (sprint always included)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    settings: List[Optional[Tuple[float, float]]] = [None] + [
        tuple(float(x) for x in s.split(":")) for s in args.settings.split(",")]
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=1) as pool:
        res = pool.map(one_setting, settings)
    for r in res:
        name = "sprint" if r["setting"] is None else f"reserve {r['setting'][0]:g} km/s, {r['setting'][1]:g} s"
        print(f"\n== {name}")
        for k in (300, 500, 800):
            print(f"  from {k} km: {r[f'flight_{k}']}")
        print(f"  vs EVASIVE from 500 km (3 seeds): {r['evade_500']}")
        for n, rows in r["pd"].items():
            print(f"  PD salvo {n}: " + "; ".join(
                f"{x['blinded']} blinded / {x['hits']} hits ({x['turrets']} turrets)" for x in rows))
        print(f"  reach abeam (km) at 100/200/300 km to go: {r['reach']}")
        print(f"  runner (burns away from launch): {r['runner']}")
        for n, rows in r["wall_pd"].items():
            print(f"  wall PD (BB + 3 FF) salvo {n}: " + "; ".join(
                f"{x['blinded']} blinded / {x['hits']} hits" for x in rows))
        for k, rows in r["wall_salvo"].items():
            print(f"  3-BC wall, 15 rounds on one ({k}): " + "; ".join(
                f"{x['hits']} hits {x['kills']} kills {x['hulk_hits']} on hulks "
                f"rt {x['retargeted']} (surplus {x['surplus']})" for x in rows))
    if args.json:
        args.json.write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
