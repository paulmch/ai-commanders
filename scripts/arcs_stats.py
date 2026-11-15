#!/usr/bin/env python3
"""
Weapon-arcs statistics over seeded free battles (heuristic captains, no LLM).

For each seed one draft battle is fought three times on the same rosters:
arcs on with roll-to-unmask, arcs on without roll, arcs off. Reported:

  PD   for every tick and every live hostile torpedo inside a ship's PD
       envelope: how many of the ship's operational turrets BEAR on it (own
       hull does not mask the line of sight) against the turret count -
       per hull class and per fleet, split into torpedoes homing on the ship
       itself and all torpedoes in its envelope. Geometry only, so it is
       measured in every mode with the same masks.
  guns rounds that fire control released and the mount held back (masked by
       the own hull / still slewing) against rounds released, and rounds
       actually fired in each mode.

    uv run python scripts/arcs_stats.py --seeds 1,2,3,4,5,6 --budget 150 --jobs 6
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Asymmetric pairings (alpha doctrine, beta doctrine), cycled over the seeds:
# every pairing has torpedoes in the air and guns in range at some point.
PAIRINGS = [("torpedo", "gunline"), ("combined", "flagship"), ("torpedo", "combined"),
            ("gunline", "flagship"), ("combined", "gunline"), ("torpedo", "flagship")]
MODES = {"arcs+roll": (True, True), "arcs, no roll": (True, False), "arcs off": (False, False)}


def run_battle(job):
    seed, budget, time_limit, mode, pair_index, mounts_path = job
    arcs_on, roll_on = MODES[mode]
    from src import weapon_arcs
    from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    from src.llm.fleet_config import BattleFleetConfig
    from src.llm.fleet_draft import auto_draft, draft_to_fleet_definition
    from src.llm.heuristic_captain import HEURISTIC_MODEL
    from src.llm.prompts import CaptainPersonality
    from src.simulation import CombatSimulation

    pairing = PAIRINGS[pair_index % len(PAIRINGS)]
    drafts = {f: auto_draft(f, budget=budget, max_ships=8, seed=seed, doctrine=d)
              for f, d in zip(("alpha", "beta"), pairing)}
    fleet_config = BattleFleetConfig(
        battle_name="arcs stats",
        alpha_fleet=draft_to_fleet_definition(
            drafts["alpha"], 500.0, captain_model=HEURISTIC_MODEL, admiral_config=None),
        beta_fleet=draft_to_fleet_definition(
            drafts["beta"], 500.0, captain_model=HEURISTIC_MODEL, admiral_config=None),
        time_limit_s=time_limit, initial_distance_km=500.0,
        record_battle=False, record_sim_trace=False,
        personality_selection=False, use_notebooks=False)
    battle_config = BattleConfig(
        initial_distance_km=500.0, time_limit_s=time_limit, verbose=False,
        personality_selection=False, record_battle=False, record_sim_trace=False,
        record_predictions=False, seed=seed)
    dummy = LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                             personality=CaptainPersonality.BALANCED)
    runner = LLMBattleRunner(config=battle_config, alpha_config=dummy, beta_config=dummy,
                             client=None, fleet_config=fleet_config)

    setup = LLMBattleRunner.setup_fleet_battle
    step = CombatSimulation.step
    table = weapon_arcs.load_hull_mounts(mounts_path)
    # (scope, hull or fleet, self-homing?) -> [torpedo-ticks, sum bearing, sum turrets, ticks with 0 bearing]
    pd = defaultdict(lambda: [0, 0, 0, 0, {}])   # ..., {turrets bearing k: ticks}
    stat_arcs = {}

    def probed_setup(self, data):
        setup(self, data)
        sim = self.simulation
        sim.echo = False
        sim.weapon_arcs = arcs_on
        sim.roll_to_unmask = roll_on
        sim.hull_mounts = table

    def probed_step(self):
        if self is runner.simulation and self.torpedoes:
            for ship in self.ships.values():
                if (ship.is_destroyed or ship.is_dying or ship.is_surrendered
                        or not ship.point_defense):
                    continue
                turrets = [(i, p) for i, p in enumerate(ship.point_defense) if p.is_operational]
                if not turrets:
                    continue
                arcs = ship.arcs
                if arcs is None or not self.weapon_arcs:
                    # arcs off: measure the same geometry on a private frame
                    arcs = stat_arcs.get(ship.ship_id)
                    if arcs is None:
                        hull = table.get(ship.ship_type)
                        arcs = stat_arcs[ship.ship_id] = weapon_arcs.ShipArcs(None, ship)
                        arcs.hull = hull
                    arcs.set_frame(ship.forward, ship.kinematic_state.up, self.current_time)
                    hull = arcs.hull
                    masks = {m.slot: m.mask for m in hull.mounts} if hull else {}
                    turret_masks = [masks.get(f"pd_{i}") for i, _ in turrets]
                else:
                    arcs.set_frame(ship.forward, ship.kinematic_state.up, self.current_time)
                    turret_masks = [p.mount.mask if p.mount is not None else None
                                    for _, p in turrets]
                envelope_m = max(p.laser.range_km for _, p in turrets) * 1000.0
                own = ship.position
                for flight in self.torpedoes:
                    if flight.is_disabled:
                        continue
                    source = self.get_ship(flight.source_ship_id)
                    if source is not None and source.faction == ship.faction:
                        continue
                    pos = flight.torpedo.position
                    dx, dy, dz = pos.x - own.x, pos.y - own.y, pos.z - own.z
                    n = (dx * dx + dy * dy + dz * dz) ** 0.5
                    if not 0.0 < n <= envelope_m:
                        continue
                    cell = weapon_arcs.cell_index(*arcs.to_body(dx / n, dy / n, dz / n))
                    bearing = sum(1 for mask in turret_masks if mask is None or not mask[cell])
                    homing = flight.torpedo.target_id == ship.ship_id
                    for key in (("hull", ship.ship_type, homing), ("fleet", ship.faction, homing)):
                        row = pd[key]
                        row[0] += 1
                        row[1] += bearing
                        row[2] += len(turrets)
                        row[3] += bearing == 0
                        row[4][bearing] = row[4].get(bearing, 0) + 1
        return step(self)

    LLMBattleRunner.setup_fleet_battle = probed_setup
    CombatSimulation.step = probed_step
    try:
        result = runner.run_battle(load_fleet_data())
    finally:
        LLMBattleRunner.setup_fleet_battle = setup
        CombatSimulation.step = step
    sim = runner.simulation
    shots = {f: sum(s.shots_fired for s in sim.ships.values() if s.faction == f)
             for f in ("alpha", "beta")}
    kills = {f: sum(1 for s in sim.ships.values() if s.faction != f and (s.is_destroyed or s.is_dying))
             for f in ("alpha", "beta")}
    rolled = sum(1 for s in sim.ships.values()
                 if s.arcs is not None and abs(s.arcs.roll) > 0.05)
    return {
        "seed": seed, "mode": mode, "pairing": list(pairing),
        "winner": result.winner, "reason": result.reason, "duration_s": result.duration_s,
        "shots": shots, "kills": kills, "tally": dict(getattr(sim, "arcs_tally", {})),
        "ships_rolled": rolled, "ships": len(sim.ships),
        "pd": [[*k, *v] for k, v in pd.items()],   # scope, name, homing, ticks, bearing, turrets, none, {k: ticks}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", default="1,2,3,4,5,6")
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--json", default=None, help="Write the raw per-battle rows here")
    parser.add_argument("--mounts", default=None,
                        help="Mount table to use instead of data/hull_mounts.json "
                             "(e.g. an earlier placement, for a before/after on one engine)")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    jobs = [(seed, args.budget, args.time_limit, mode, i, args.mounts)
            for i, seed in enumerate(seeds) for mode in MODES]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run_battle, jobs))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))

    try:
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%h %s", "--", "data/hull_mounts.json"],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent).stdout.strip()
    except OSError:
        commit = "?"
    print(f"mount table: {args.mounts or 'data/hull_mounts.json'} ({'given file' if args.mounts else commit})")
    print(f"{len(seeds)} seeds x {len(MODES)} modes, budget {args.budget}, "
          f"time limit {args.time_limit:.0f} s\n")

    print("BATTLES")
    for row in rows:
        t = row["tally"]
        print(f"  seed {row['seed']} {'/'.join(row['pairing']):<18} {row['mode']:<14} "
              f"winner {str(row['winner'] or 'draw'):<5} {row['reason']:<24} {row['duration_s']:>4.0f} s  "
              f"kills a/b {row['kills']['alpha']}/{row['kills']['beta']}  "
              f"shots a/b {row['shots']['alpha']}/{row['shots']['beta']}  "
              f"released {t.get('released', 0)} masked {t.get('masked', 0)} "
              f"slewing {t.get('slewing', 0)}  rolled {row['ships_rolled']}/{row['ships']}")

    print("\nGUNS (summed over the seeds)")
    for mode in MODES:
        sel = [r for r in rows if r["mode"] == mode]
        shots = sum(r["shots"]["alpha"] + r["shots"]["beta"] for r in sel)
        rel = sum(r["tally"].get("released", 0) for r in sel)
        masked = sum(r["tally"].get("masked", 0) for r in sel)
        slewing = sum(r["tally"].get("slewing", 0) for r in sel)
        held = (f"ready-ticks (a mount cleared to fire by fire control, counted every tick) "
                f"{rel}: held masked {masked} ({100 * masked / rel:.1f}%), "
                f"held slewing {slewing} ({100 * slewing / rel:.1f}%)"
                if rel else "no gate")
        print(f"  {mode:<14} rounds fired {shots:>5}   {held}")

    for mode in MODES:
        print(f"\nPD BEARING, {mode} (torpedo-ticks inside the PD envelope; "
              f"bearing = turrets whose own hull leaves the line of sight open)")
        agg = defaultdict(lambda: [0, 0, 0, 0, defaultdict(int)])
        for r in rows:
            if r["mode"] != mode:
                continue
            for scope, name, homing, n, bearing, turrets, zero, hist in r["pd"]:
                for key in ((scope, name, "homing on the ship") if homing else None,
                            (scope, name, "all in envelope")):
                    if key is None:
                        continue
                    a = agg[key]
                    a[0] += n
                    a[1] += bearing
                    a[2] += turrets
                    a[3] += zero
                    for k, ticks in hist.items():
                        a[4][int(k)] += ticks
        print(f"  {'scope':<6}{'hull / fleet':<19}{'torpedoes':<20}{'ticks':>7}"
              f"{'bearing/turrets':>17}   share of ticks with k turrets bearing")
        for (scope, name, which), (n, bearing, turrets, zero, hist) in sorted(agg.items()):
            shares = "  ".join(f"k={k}: {100 * hist[k] / n:.0f}%" for k in sorted(hist))
            print(f"  {scope:<6}{name:<19}{which:<20}{n:>7}"
                  f"{(100 * bearing / turrets if turrets else 0):>16.1f}%   {shares}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
