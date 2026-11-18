#!/usr/bin/env python3
"""
Launch-sequence measurements (src/torpedo_launch.py): seconds from launch to
ignition, how close each round comes to its own hull, whether it sits in its
ship's drive plume, and how far the rounds of one salvo stay apart - per
launcher hull and launch geometry, on the real engine (PD off, target a
battlecruiser holding course, no RNG involved in the sequence).

Geometries (the target 500 km away, on the shooter's +x unless stated):
  coast_ahead     shooter coasting, nose on the target
  coast_abeam     shooter coasting, target 90 deg off the nose (port)
  coast_astern    shooter coasting, target dead astern
  burn_toward     full throttle at the target (nose on it)
  burn_away       full throttle straight away from it (target astern)
  turning         a full-throttle HEADING order 120 deg round started 3 s
                  before launch (a thrust-vectored turn under way at launch)
  spinning        coasting, spinning at its hull's peak thrust-vectored rate
                  about the dorsal axis (yaw toward port) at launch
  pair_formation  two of these hulls 120 m apart abeam, both launch at once
                  (friendly clearance)
  close_50km      coasting, target 50 km ahead

    uv run python scripts/launch_sequence_facts.py [--json OUT]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.physics import Vector3D  # noqa: E402
from src.simulation import (CombatSimulation, Maneuver, ManeuverType,  # noqa: E402
                            SimulationEventType, create_ship_from_fleet_data)
from src import torpedo_launch as TL  # noqa: E402
from src.drive_plume import PLUME_MAX_HALF_ANGLE_DEG  # noqa: E402

FLEET = json.loads((ROOT / "data" / "fleet_ships.json").read_text())
HULLS = ("corvette", "cruiser_torpedo")
GEOMETRIES = ("coast_ahead", "coast_abeam", "coast_astern", "burn_toward", "burn_away",
              "turning", "spinning", "pair_formation", "close_50km")
# peak thrust-vectored yaw rate (deg/s) per hull: 90 deg in the TV time,
# accelerate-then-brake, so the peak is twice the mean (CLAUDE.md table)
PEAK_RATE_DEG_S = {"corvette": 2 * 90 / 12.1, "cruiser_torpedo": 2 * 90 / 28.2}


def _ship(sid, cls, faction, pos, fwd, vel=(0.0, 0.0, 0.0)):
    return create_ship_from_fleet_data(sid, cls, faction, FLEET, position=Vector3D(*pos),
                                       velocity=Vector3D(*vel), forward=Vector3D(*fwd))


def _heading(throttle, d):
    return Maneuver(maneuver_type=ManeuverType.HEADING, start_time=0.0, duration=1e9,
                    throttle=throttle, heading_direction={"x": d[0], "y": d[1], "z": d[2]})


def scenario(cls: str, geometry: str, sequence: bool = True, horizon_s: float = 40.0):
    sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=7)
    sim.echo = False
    sim.launch_sequence = sequence
    rng = 50_000.0 if geometry == "close_50km" else 500_000.0
    tpos = {"coast_abeam": (0.0, rng, 0.0), "coast_astern": (-rng, 0.0, 0.0),
            "burn_away": (-rng, 0.0, 0.0)}.get(geometry, (rng, 0.0, 0.0))
    shooters = [_ship("sh", cls, "alpha", (0, 0, 0), (1, 0, 0))]
    if geometry == "pair_formation":
        shooters.append(_ship("sh2", cls, "alpha", (0, 120.0, 0), (1, 0, 0)))
    tg = _ship("tg", "battlecruiser", "beta", tpos, (-1, 0, 0))
    tg.point_defense = []
    for s in shooters:
        s.point_defense = []
        sim.add_ship(s)
    sim.add_ship(tg)
    pre = 0
    if geometry in ("burn_toward",):
        shooters[0].current_maneuver = _heading(1.0, (1, 0, 0))
        pre = 3
    elif geometry == "burn_away":
        shooters[0].current_maneuver = _heading(1.0, (1, 0, 0))
        pre = 3
    elif geometry == "turning":
        c, s_ = math.cos(math.radians(120)), math.sin(math.radians(120))
        shooters[0].current_maneuver = _heading(1.0, (c, s_, 0.0))
        pre = 3
    elif geometry == "spinning":
        shooters[0].angular_velocity_rad_s = math.radians(PEAK_RATE_DEG_S[cls])
        shooters[0].angular_axis = Vector3D(0.0, 0.0, 1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(pre):
            sim.step()
        n = 4 if cls == "cruiser_torpedo" else 1
        for s in shooters:
            for _ in range(n):
                sim._handle_torpedo_command(s, {"target_id": "tg"})
    rounds = {tf.torpedo_id: {"src": tf.source_ship_id, "t0": sim.current_time,
                              "ign": None, "reason": None, "min_gap_seq": math.inf,
                              "min_gap_after": math.inf, "plume_s": 0.0, "min_gap_friend": math.inf,
                              "min_gap_ign": None, "impact_t": None,
                              "forecast": tf.torpedo.launch_seq.hold_forecast_s
                              if tf.torpedo.launch_seq is not None else 0.0}
              for tf in sim.torpedoes}
    min_pair = math.inf
    t_launch = sim.current_time
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.current_time < t_launch + horizon_s and sim.torpedoes:
            events = sim.step()
            for e in events:
                if e.event_type == SimulationEventType.TORPEDO_IGNITION:
                    r = rounds[e.data["torpedo_id"]]
                    r["ign"] = e.data["delay_s"]
                    r["reason"] = e.data["reason"]
                    r["min_gap_ign"] = e.data["own_gap_m"]
                if e.event_type == SimulationEventType.TORPEDO_IMPACT:
                    rounds[e.data["torpedo_id"]]["impact_t"] = e.timestamp - t_launch
            live = sim.torpedoes
            for tf in live:
                r = rounds[tf.torpedo_id]
                own = sim.get_ship(tf.source_ship_id)
                snap = TL.ShipSnap(own)
                gap = TL.hull_gap_m(snap, tf.torpedo.position)
                key = "min_gap_seq" if tf.torpedo.in_launch_sequence else "min_gap_after"
                if sim.current_time - t_launch >= 1.0 - 1e-9:
                    r[key] = min(r[key], gap)
                for other in sim.ships.values():
                    if other.faction == own.faction and other is not own:
                        r["min_gap_friend"] = min(r["min_gap_friend"], TL.hull_gap_m(
                            TL.ShipSnap(other), tf.torpedo.position))
                # in its own ship's drive plume while that drive burns?
                if getattr(own, "applied_throttle", 0.0) > 0.0:
                    tail = own.position - own.forward * snap.half_len
                    rel = tf.torpedo.position - tail
                    if rel.magnitude > 0 and rel.magnitude < 100_000:
                        ang = math.degrees(TL.angle_between(own.forward * -1.0, rel))
                        if ang < PLUME_MAX_HALF_ANGLE_DEG and tf.torpedo.in_launch_sequence:
                            r["plume_s"] += 1.0
            if len(live) > 1 and sim.current_time - t_launch <= 15:
                for i in range(len(live)):
                    for j in range(i + 1, len(live)):
                        min_pair = min(min_pair, live[i].torpedo.position.distance_to(
                            live[j].torpedo.position))
    return rounds, (None if min_pair == math.inf else min_pair)


def table(sequence: bool = True):
    out = {}
    for cls in HULLS:
        for g in GEOMETRIES:
            rounds, pair = scenario(cls, g, sequence)
            ign = [r["ign"] for r in rounds.values() if r["ign"] is not None]
            out[f"{cls}/{g}"] = {
                "ignition_s": ign,
                "forecast_s": [r["forecast"] for r in rounds.values()],
                "reasons": sorted({r["reason"] for r in rounds.values() if r["reason"]}),
                "min_gap_during_seq_m": round(min(r["min_gap_seq"] for r in rounds.values()), 1),
                "min_gap_after_ignition_m": round(min(r["min_gap_after"] for r in rounds.values()), 1),
                "plume_seconds": sum(r["plume_s"] for r in rounds.values()),
                "min_gap_friendly_hull_m": (None if all(r["min_gap_friend"] == math.inf for r in rounds.values())
                                            else round(min(r["min_gap_friend"] for r in rounds.values()), 1)),
                "min_round_pair_m": None if pair is None else round(pair, 1),
                "impact_s": [None if r["impact_t"] is None else round(r["impact_t"], 1)
                             for r in rounds.values()],
            }
    return out


# The cases behind the doctrine's LAUNCH_FACT (src/llm/torpedo_doctrine.py),
# coasting launcher: "ahead" = the slowest of the rounds that need at most a
# quarter turn onto the target, "half_round" = the slowest of those that must
# turn about (written to tests/fixtures/torpedo_link_facts.json "launch" by
# scripts/torpedo_link_facts.py).
AHEAD_CASES = ("corvette/coast_ahead", "corvette/coast_abeam", "cruiser_torpedo/coast_ahead")
HALF_ROUND_CASES = ("corvette/coast_astern", "cruiser_torpedo/coast_abeam")


def doctrine_facts() -> dict:
    res = table()
    return {
        "ahead_s": max(max(res[c]["ignition_s"]) for c in AHEAD_CASES),
        "half_round_s": max(max(res[c]["ignition_s"]) for c in HALF_ROUND_CASES),
        "cases": {c: res[c]["ignition_s"] for c in AHEAD_CASES + HALF_ROUND_CASES},
    }


def forecasts(sequence: bool = True):
    """Time of flight forecast against flown, at launch, for a target holding
    course (battlecruiser, PD off): the captain's shadow flight (fly_torpedo,
    the TOF of the fire solution and its PROFILES line), the admiral's 1-D
    pursuit (fleet_intel.solve_pursuit, target holding course, with the
    forecast hold) and the datalink board (torpedo_link.round_forecast)."""
    from src.llm.fire_solutions import fly_torpedo
    from src.llm import fleet_intel as intel
    from src import torpedo_link as L
    out = {}
    for cls in HULLS:
        for rng_km in (50.0, 150.0, 300.0, 500.0):
            for vc in (0.0, 3.0):
                for prof in ("sprint", "reserve"):
                    def build():
                        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=7)
                        sim.echo = False
                        sim.launch_sequence = sequence
                        sh = _ship("sh", cls, "alpha", (0, 0, 0), (1, 0, 0), vel=(vc * 1000, 0, 0))
                        tg = _ship("tg", "battlecruiser", "beta", (rng_km * 1000, 0, 0), (-1, 0, 0))
                        for s in (sh, tg):
                            s.point_defense = []
                            sim.add_ship(s)
                        return sim, sh, tg
                    sim, sh, tg = build()
                    with contextlib.redirect_stdout(io.StringIO()):
                        shot = fly_torpedo(sim, sh, tg, FLEET, target_evades=False, profile=prof)
                    specs = sh.ready_torpedo_launchers[0].specs
                    hold = TL.predict_launch_hold_s(sim, sh, tg, specs)
                    fly = intel.solve_pursuit(rng_km * 1000, vc * 1000, 0.0, 0.0, specs,
                                              hold_s=hold, profile=prof)
                    cmd = {"target_id": "tg"}
                    if prof != "sprint":
                        cmd["profile"] = prof
                    sim._handle_torpedo_command(sh, cmd)
                    tf = sim.torpedoes[-1]
                    board = L.round_forecast(tf.torpedo, tg).eta_s
                    t0, flown = sim.current_time, None
                    with contextlib.redirect_stdout(io.StringIO()):
                        while sim.torpedoes and sim.current_time < t0 + 300 and flown is None:
                            for e in sim.step():
                                if e.event_type == SimulationEventType.TORPEDO_IMPACT:
                                    flown = sim.current_time - t0
                    out[f"{cls}/{rng_km:.0f}km/vc{vc:g}/{prof}"] = {
                        "flown_s": flown, "captain_s": shot.tof_s if shot and shot.hits else None,
                        "admiral_s": fly.get("tof_s"), "board_s": round(board, 1),
                        "hold_forecast_s": hold,
                        "captain_gj": round(shot.impact_gj, 1) if shot and shot.hits else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the table here")
    ap.add_argument("--no-sequence", action="store_true", help="old instant ignition")
    ap.add_argument("--forecasts", action="store_true",
                    help="forecast vs flown time of flight instead of the geometry table")
    args = ap.parse_args()
    if args.forecasts:
        res = forecasts(not args.no_sequence)
        print(f"{'case':40s} {'flown':>6s} {'capt':>6s} {'adm':>6s} {'board':>6s} {'hold':>5s} {'GJ':>5s}")
        for k, v in res.items():
            f = lambda x: "-" if x is None else f"{x:.0f}"  # noqa: E731
            print(f"{k:40s} {f(v['flown_s']):>6s} {f(v['captain_s']):>6s} {f(v['admiral_s']):>6s} "
                  f"{f(v['board_s']):>6s} {v['hold_forecast_s']:5.0f} {f(v['captain_gj']):>5s}")
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=1))
        return
    res = table(not args.no_sequence)
    print(f"{'hull/geometry':34s} {'ign s':>12s} {'fcst s':>8s} {'reason':>18s} {'gap seq':>8s} "
          f"{'gap aft':>8s} {'plume':>5s} {'pair':>6s} {'friend':>6s} impact s")
    for k, v in res.items():
        ign = ",".join(f"{x:g}" for x in sorted(set(v["ignition_s"]))) or "-"
        fc = ",".join(f"{x:g}" for x in sorted(set(v["forecast_s"]))) or "-"
        print(f"{k:34s} {ign:>12s} {fc:>8s} {'/'.join(v['reasons']) or '-':>18s} "
              f"{v['min_gap_during_seq_m']:8.1f} {v['min_gap_after_ignition_m']:8.1f} "
              f"{v['plume_seconds']:5.0f} {str(v['min_round_pair_m']):>6s} "
              f"{str(v['min_gap_friendly_hull_m']):>6s} "
              f"{','.join(str(x) for x in v['impact_s'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
