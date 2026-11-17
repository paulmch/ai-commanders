#!/usr/bin/env python3
"""
Torpedo retarget audit: for every round in a recording whose target died in
flight, replay the engine's own retarget rule on the recorded state and say
exactly why it did or did not swing onto another enemy.

The engine evaluates `CombatSimulation._retarget_torpedo` on every tick a
round with a live seeker and fuel is orphaned. In step T (events stamped T)
the ships have already moved to their end-of-step state (trace frame T+1)
while the round is still at its start-of-step state (frame T). This script
rebuilds exactly that: one CombatSimulation holding every hull of the battle
(created from fleet data by class, put at the recorded position/velocity,
destroyed / dying flags from the frame), the round as an engine `Torpedo` +
`TorpedoInFlight` (corvette/torpedo-cruiser launcher specs, mass from the
recorded delta-v by the rocket equation, PD heat from the frame), and then
calls the engine:

- `sim._retarget_torpedo(flight)` -> what the engine would pick (if anything);
- `torpedo.can_intercept(pos, vel)` per live enemy -> reachable?, time, fuel
  left after the steer. The delta-v the steer NEEDS is read from the same
  function on a copy holding an unlimited tank (same mass), so no maths is
  re-implemented here.

Each candidate gets the gate that refused it, in the engine's order:
`no_solution` (target out-accelerates the round, t = inf), `dv` (steer needs
more than the tank), `time` (> RETARGET_MAX_INTERCEPT_S), `reserve` (fuel left
after the steer below the reserve), or `ok`.

Alternative rules are evaluated on the same candidates (counterfactual, the
engine is not changed): reserve 0; a reserve scaled to the time to go
(`scaled`: min(engine reserve, max(floor, 0.5 * a_max(candidate) * t)) -
the lateral delta-v a target burning flat out across the line of sight can
force in t seconds).

`--fly` additionally flies selected (round, candidate) pairs in a fresh
seeded CombatSimulation with PD off from the recorded state: the round is
pointed at the candidate the way the engine does it, every hull coasts on
its automatic rule-based layer (no orders), and the result (hit / miss and
closest approach) is compared with can_intercept's verdict. `--fly-burn`
gives the candidate a HEADING order along its recorded nose at its recorded
throttle instead (what it was doing at the time).

Usage:
    uv run python scripts/torpedo_retarget_audit.py RECORDING [--json OUT]
        [--fly] [--fly-burn] [--max-steps N]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.battle_runner import load_fleet_data  # noqa: E402
from src.physics import Vector3D  # noqa: E402
from src.recording_io import load_recording  # noqa: E402
from src.simulation import (  # noqa: E402
    CombatSimulation,
    Maneuver,
    ManeuverType,
    SimulationEventType,
    TorpedoInFlight,
    create_ship_from_fleet_data,
)
from src.torpedo import GuidanceMode, Torpedo  # noqa: E402

G = 9.81


def _v(a) -> Vector3D:
    return Vector3D(float(a[0]), float(a[1]), float(a[2]))


def engine_reserve_kps(sim: CombatSimulation, torp: Torpedo) -> float:
    return max(sim.RETARGET_MIN_RESERVE_KPS,
               torp.specs.total_delta_v_kps * sim.RETARGET_RESERVE_FRACTION)


def scaled_reserve_kps(engine_reserve: float, cand_accel_mps2: float,
                       t_s: float, floor_kps: float = 0.1) -> float:
    """Counterfactual: the lateral delta-v a candidate burning flat out across
    the line of sight can force in t seconds (0.5 * a * t), floored, never
    above the engine's reserve."""
    return min(engine_reserve, max(floor_kps, 0.5 * cand_accel_mps2 * t_s / 1000.0))


def make_round(specs, pos, vel, target_id, dv_kps, heat_j, source_id, tid,
               launch_time=0.0) -> TorpedoInFlight:
    """An engine round in a recorded state (mass from dv by the rocket eq.)."""
    torp = Torpedo(specs=specs, position=pos, velocity=vel, target_id=target_id,
                   guidance_mode=GuidanceMode.COLLISION)
    torp.remaining_delta_v_kps = dv_kps
    torp.current_mass_kg = specs.dry_mass_kg * math.exp(
        dv_kps / specs.exhaust_velocity_kps)
    torp.armed = True
    torp.fuel_exhausted = dv_kps <= 0.0
    flight = TorpedoInFlight(torpedo_id=tid, torpedo=torp,
                             source_ship_id=source_id, launch_time=launch_time)
    flight.heat_absorbed_j = heat_j
    return flight


def steer_needed_kps(torp: Torpedo, pos: Vector3D, vel: Vector3D) -> tuple:
    """(t_intercept, dv the steer needs) from the engine's can_intercept on a
    copy of the round with an unlimited tank at the same mass."""
    probe = copy.copy(torp)
    probe.remaining_delta_v_kps = 1.0e6
    ok, t, left = probe.can_intercept(pos, vel)
    if not ok:
        return t, float('inf')
    return t, 1.0e6 - left


def point_round_at(sim: CombatSimulation, flight: TorpedoInFlight, ship_id: str):
    """Point a round at a ship exactly the way _retarget_torpedo does after
    its choice (target id, APN estimator and CPA / latch reset)."""
    torp = flight.torpedo
    torp.target_id = ship_id
    torp.last_observed_target_velocity = None
    torp.estimated_target_accel = Vector3D.zero()
    flight.min_distance_to_target = float('inf')
    flight.prev_distance_to_target = float('inf')
    flight.terminal_latched = False


@dataclass
class Battle:
    rec: dict
    frames: dict
    ship_types: dict
    factions: dict
    events: list
    fleet_data: dict = field(default_factory=load_fleet_data)

    @classmethod
    def load(cls, path: str) -> 'Battle':
        rec = load_recording(path)
        frames = {round(f['t'], 3): f for f in rec['sim_trace']}
        ship_types, factions = {}, {}
        for side in ('alpha', 'beta'):
            for s in rec[f'{side}_fleet']['ships']:
                ship_types[s['ship_id']] = s['ship_type']
                factions[s['ship_id']] = side
        return cls(rec, frames, ship_types, factions, rec['events'])

    def specs_for(self, ship_id: str):
        ship = create_ship_from_fleet_data(
            ship_id, self.ship_types[ship_id], self.factions[ship_id],
            self.fleet_data, position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
        return ship.torpedo_launchers[0].specs

    def build_sim(self, frame: dict, seed: int = 1) -> CombatSimulation:
        random.seed(seed)
        sim = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=seed)
        for sid, st in frame['ships'].items():
            fwd = _v(st.get('fwd', [1, 0, 0]))
            ship = create_ship_from_fleet_data(
                sid, self.ship_types[sid], self.factions[sid], self.fleet_data,
                position=_v(st['pos']), velocity=_v(st['vel']), forward=fwd)
            ship.is_destroyed = bool(st.get('destroyed'))
            ship.is_dying = bool(st.get('dying'))
            sim.add_ship(ship)
        sim.current_time = frame['t']
        return sim

    def set_ships(self, sim: CombatSimulation, frame: dict) -> None:
        for sid, st in frame['ships'].items():
            ship = sim.get_ship(sid)
            ship.kinematic_state.position = _v(st['pos'])
            ship.kinematic_state.velocity = _v(st['vel'])
            ship.is_destroyed = bool(st.get('destroyed'))
            ship.is_dying = bool(st.get('dying'))
        sim.current_time = frame['t']


def ship_accel_mps2(sim: CombatSimulation, ship_id: str) -> float:
    ship = sim.get_ship(ship_id)
    return ship.kinematic_state.max_acceleration_g() * G


def orphan_episodes(b: Battle) -> list:
    """Every (round, dead target, step of death) where the round was still
    in flight after the step its target died in."""
    deaths = {}
    for e in b.events:
        if e.get('event_type') == 'ship_destroyed':
            deaths[e['ship_id']] = e['timestamp']
    eps = []
    for sid, td in sorted(deaths.items(), key=lambda kv: kv[1]):
        f0, f1 = b.frames.get(td), b.frames.get(td + 1.0)
        if f0 is None or f1 is None:
            continue
        alive_after = {t['id'] for t in f1['torpedoes']}
        for t in f0['torpedoes']:
            if t['target'] == sid and t['id'] in alive_after:
                eps.append({'torpedo_id': t['id'], 'dead_target': sid,
                            't_death': td, 'source': t['source']})
    return eps


def evaluate_step(b: Battle, sim: CombatSimulation, specs, ep: dict, T: float,
                  tf0: dict, tf1: dict) -> dict:
    """One engine retarget evaluation: round at frame T, ships at frame T+1."""
    f1 = b.frames[T + 1.0]
    b.set_ships(sim, f1)
    flight = make_round(specs, _v(tf0['pos']), _v(tf0['vel']), ep['dead_target'],
                        tf0['dv_remaining'], tf1.get('pd_heat_j', 0.0),
                        ep['source'], ep['torpedo_id'])
    flight.is_disabled = bool(tf1.get('disabled'))
    torp = flight.torpedo
    reserve = engine_reserve_kps(sim, torp)
    dead = sim.get_ship(ep['dead_target'])
    rel = dead.position - torp.position
    dist_dead_km = rel.magnitude / 1000.0
    closing = -(dead.velocity - torp.velocity).dot(rel.normalized())
    cands = []
    for cand in sim.get_enemy_ships(ep['source']):
        ok, t, left = torp.can_intercept(cand.position, cand.velocity)
        t_need, dv_need = steer_needed_kps(torp, cand.position, cand.velocity)
        if math.isinf(t_need):
            gate = 'no_solution'
        elif not ok:
            gate = 'dv'
        elif t > sim.RETARGET_MAX_INTERCEPT_S:
            gate = 'time'
        elif left < reserve:
            gate = 'reserve'
        else:
            gate = 'ok'
        a = ship_accel_mps2(sim, cand.ship_id)
        sres = scaled_reserve_kps(reserve, a, t_need)
        reach = (not math.isinf(t_need) and dv_need <= torp.remaining_delta_v_kps
                 and t_need <= sim.RETARGET_MAX_INTERCEPT_S)
        cands.append({
            'ship': cand.ship_id, 'class': cand.ship_type,
            'range_km': round(torp.position.distance_to(cand.position) / 1000, 1),
            't_s': None if math.isinf(t_need) else round(t_need, 1),
            'dv_need': None if math.isinf(dv_need) else round(dv_need, 3),
            'can_intercept': ok, 'fuel_left': round(left, 3) if ok else None,
            'gate': gate,
            'ok_reserve0': reach,
            'scaled_reserve': round(sres, 3),
            'ok_scaled': reach and (torp.remaining_delta_v_kps - dv_need) >= sres,
        })
    pick = None
    if not flight.is_disabled and not torp.fuel_exhausted:
        chosen = sim._retarget_torpedo(flight)
        pick = chosen.ship_id if chosen is not None else None
    return {
        'T': T, 'dv': tf0['dv_remaining'], 'disabled': flight.is_disabled,
        'heat_frac': round(flight.heat_absorbed_j / flight.ELECTRONICS_THRESHOLD_J, 3),
        'reserve': round(reserve, 3),
        'dist_dead_km': round(dist_dead_km, 1),
        's_to_dead': round(dist_dead_km * 1000 / closing, 1) if closing > 0 else None,
        'engine_pick': pick, 'candidates': cands,
    }


def audit(b: Battle, max_steps: int = 120) -> list:
    """Every orphan episode, evaluated from the death step until the round
    retargets, is blinded, leaves, or max_steps ticks pass."""
    retargets = {}
    for e in b.events:
        if e.get('event_type') == 'torpedo_retargeted':
            retargets.setdefault(e['data']['torpedo_id'], []).append(
                (e['timestamp'], e['data']['old_target_id'], e['data']['new_target_id']))
    ends = {}
    for e in b.events:
        if e.get('event_type') in ('torpedo_miss', 'torpedo_impact'):
            ends[e['data'].get('torpedo_id') or e['data'].get('projectile_id')] = (
                e['timestamp'], e['event_type'], e['data'].get('reason'),
                e['data'].get('target_id'))
    sim = None
    out = []
    specs_cache = {}
    for ep in orphan_episodes(b):
        if sim is None:
            sim = b.build_sim(b.frames[ep['t_death'] + 1.0])
        src = ep['source']
        if src not in specs_cache:
            specs_cache[src] = b.specs_for(src)
        specs = specs_cache[src]
        rt = [r for r in retargets.get(ep['torpedo_id'], [])
              if r[1] == ep['dead_target'] and r[0] >= ep['t_death']]
        steps = []
        T = ep['t_death']
        for _ in range(max_steps):
            f0, f1 = b.frames.get(T), b.frames.get(T + 1.0)
            if f0 is None or f1 is None:
                break
            tf0 = next((t for t in f0['torpedoes'] if t['id'] == ep['torpedo_id']), None)
            tf1 = next((t for t in f1['torpedoes'] if t['id'] == ep['torpedo_id']), None)
            if tf0 is None or tf1 is None:
                break
            if tf0['target'] != ep['dead_target']:
                break  # retargeted in an earlier step
            ev = evaluate_step(b, sim, specs, ep, T, tf0, tf1)
            steps.append(ev)
            if ev['disabled'] or tf1['target'] != ep['dead_target']:
                break
            T += 1.0
        ep = dict(ep)
        ep['recorded_retarget'] = rt
        ep['end'] = ends.get(ep['torpedo_id'])
        ep['steps'] = steps
        out.append(ep)
    return out


# ---------------------------------------------------------------------------
# Flying the verdicts
# ---------------------------------------------------------------------------

def fly(b: Battle, ep: dict, T: float, cand_id: str, burn: bool = False,
        seed: int = 7, horizon_s: float = 300.0) -> dict:
    """Fly the round from its frame-(T+1) state at cand_id in a fresh seeded
    sim with PD off and every hull on its rule layer only (no orders)."""
    f1 = b.frames[T + 1.0]
    sim = b.build_sim(f1, seed=seed)
    for s in sim.ships.values():
        s.point_defense = []
    tf1 = next(t for t in f1['torpedoes'] if t['id'] == ep['torpedo_id'])
    specs = b.specs_for(ep['source'])
    flight = make_round(specs, _v(tf1['pos']), _v(tf1['vel']), cand_id,
                        tf1['dv_remaining'], 0.0, ep['source'], ep['torpedo_id'],
                        launch_time=f1['t'])
    point_round_at(sim, flight, cand_id)
    sim.torpedoes.append(flight)
    if burn:
        st = f1['ships'][cand_id]
        thr = float(st.get('thrust') or 0.0)
        if thr > 0:
            fwd = st.get('fwd', [1, 0, 0])
            sim.inject_command(cand_id, Maneuver(
                ManeuverType.HEADING, start_time=f1['t'], throttle=thr,
                heading_direction={'x': fwd[0], 'y': fwd[1], 'z': fwd[2]}))
    events = []
    sim.add_event_callback(events.append)
    t_end = f1['t'] + horizon_s
    while sim.torpedoes and sim.current_time < t_end:
        sim.step()
    hit = [e for e in events if e.event_type == SimulationEventType.TORPEDO_IMPACT]
    miss = [e for e in events if e.event_type == SimulationEventType.PROJECTILE_MISS
            and e.data.get('type') == 'torpedo']
    return {
        'hit': bool(hit),
        'hit_ship': hit[0].target_id if hit else None,
        'miss_cpa_km': round(miss[0].data.get('closest_approach_km', -1), 3) if miss else None,
        'miss_reason': miss[0].data.get('reason') if miss else None,
        'flight_s': round(sim.current_time - f1['t'], 1),
        'dv_left': round(flight.torpedo.remaining_delta_v_kps, 3),
    }


def first_step_table(audited: list) -> list:
    rows = []
    for ep in audited:
        if not ep['steps']:
            continue
        s0 = ep['steps'][0]
        rows.append((ep, s0))
    return rows


def print_report(audited: list) -> None:
    for ep in audited:
        steps = ep['steps']
        if not steps:
            continue
        s0 = steps[0]
        rt = ep['recorded_retarget']
        tag = (f"RETARGETED {rt[0][2]} @T+{rt[0][0]:.0f}" if rt else
               ('BLIND' if s0['disabled'] else 'NO RETARGET'))
        print(f"\n{ep['torpedo_id']}  dead={ep['dead_target']} @T+{ep['t_death']:.0f}"
              f"  dv={s0['dv']:.2f}  heat={s0['heat_frac']:.2f}  reserve={s0['reserve']:.2f}"
              f"  {s0['dist_dead_km']:.0f} km / {s0['s_to_dead']} s to dead target"
              f"  -> {tag}; engine replay pick: {s0['engine_pick']}  end={ep['end']}")
        for c in s0['candidates']:
            print(f"    {c['ship']:8s} {c['class']:13s} {c['range_km']:7.1f} km  t={c['t_s']}"
                  f"  needs {c['dv_need']}  can_int={c['can_intercept']}  left={c['fuel_left']}"
                  f"  gate={c['gate']}  r0={c['ok_reserve0']}  scaled({c['scaled_reserve']})={c['ok_scaled']}")
        # later steps: did anything change?
        later = [s for s in steps[1:] if any(c['gate'] == 'ok' for c in s['candidates'])
                 or s['engine_pick']]
        if later:
            print(f"    later step T+{later[0]['T']:.0f}: pick={later[0]['engine_pick']}")
        best_r0 = None
        for s in steps:
            if any(c['ok_reserve0'] for c in s['candidates']):
                best_r0 = s['T']
                break
        print(f"    steps evaluated: {len(steps)}; first step any candidate reachable at reserve 0: {best_r0}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording')
    ap.add_argument('--json', help='write the full audit here')
    ap.add_argument('--max-steps', type=int, default=120)
    ap.add_argument('--fly', action='store_true',
                    help='fly every first-step (round, candidate) pair')
    ap.add_argument('--fly-burn', action='store_true',
                    help='with --fly: candidates keep their recorded burn')
    args = ap.parse_args()
    b = Battle.load(args.recording)
    audited = audit(b, args.max_steps)
    print_report(audited)
    if args.fly:
        print('\n=== flown (PD off, first orphan step, every candidate) ===')
        for ep in audited:
            if not ep['steps'] or ep['steps'][0]['disabled']:
                continue
            s0 = ep['steps'][0]
            for c in s0['candidates']:
                if c['t_s'] is None:
                    continue
                res = fly(b, ep, s0['T'], c['ship'], burn=args.fly_burn)
                c['flown'] = res
                print(f"{ep['torpedo_id']} -> {c['ship']:8s} gate={c['gate']:8s} "
                      f"dv_need={c['dv_need']} left={c['fuel_left']}  flown: {res}")
    if args.json:
        Path(args.json).write_text(json.dumps(audited, indent=1, default=str))


if __name__ == '__main__':
    main()
