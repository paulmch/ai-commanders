#!/usr/bin/env python3
"""
Side-symmetry probe: run one mirror draft battle (identical auto-drafted
fleets, identical rule-based captains, no LLM) in process and report

  * the first ticks on which the two fleets stop being mirror images of each
    other (beta is alpha rotated 180 degrees about Z), with what differs and
    which events the tick held - a break on a tick with no hit, damage roll
    or evasion jink is a deterministic asymmetry in the engine;
  * the result, so the same seed can be compared across the probes below.

Probes (same seed, same fleets):
  --beta-first   add beta's ships to the simulation before alpha's. If the
                 result swaps sides with it, the advantage follows insertion
                 (processing) order.
  --flip         rotate the whole battlefield 180 degrees about Z before the
                 first step (alpha starts at +X). If the result changes, the
                 advantage follows the geometry (an absolute-axis effect).

  --flat-rng     every random draw returns the same value (battle RNG, damage
                 stream, per-round / per-torpedo / per-ship streams), so both
                 sides get identical rolls: the battle must then stay an
                 exact mirror image to the end, and ANY break is the engine
                 or the captains treating the sides differently. Takes an
                 optional value (--flat-rng 0.05): different values walk
                 different branches (0.05: every hit, crit and fitting
                 strike lands; 0.9: almost none).

    uv run python scripts/mirror_probe.py --seed 102 --budget 200
    uv run python scripts/mirror_probe.py --seed 102 --budget 200 --beta-first
    uv run python scripts/mirror_probe.py --seed 102 --budget 200 --flat-rng

The exact, RNG-free version of this check runs in CI:
tests/test_side_symmetry.py.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.fleet_config import BattleFleetConfig
from src.llm.fleet_draft import auto_draft, draft_to_fleet_definition
from src.llm.heuristic_captain import DEFAULT_DOCTRINE, DOCTRINES, HEURISTIC_MODEL
from src.llm.prompts import CaptainPersonality
from src.physics import Vector3D
from src.simulation import CombatSimulation


class FlatRandom(random.Random):
    """Every draw returns the same value (see --flat-rng)."""

    value = 0.31

    def random(self):
        return self.value

    def getrandbits(self, k):
        return 0

    def gauss(self, mu=0.0, sigma=1.0):
        return mu

    def normalvariate(self, mu=0.0, sigma=1.0):
        return mu


def _other(ship_id):
    side, _, n = ship_id.partition("_")
    return ("beta_" if side == "alpha" else "alpha_") + n


def _mirror(v):
    return Vector3D(-v.x, -v.y, v.z)


def _close(p, q, tol):
    return all(abs(x - y) <= tol * max(1.0, abs(x), abs(y)) for x, y in zip(p, q))


def _xyz(v):
    return (v.x, v.y, v.z)


def mirror_breaks(sim, tol):
    """Differences between alpha and the mirror image of beta, as text."""
    out = []
    for sid, a in sim.ships.items():
        if a.faction != "alpha" or _other(sid) not in sim.ships:
            continue
        b = sim.ships[_other(sid)]
        for name in ("position", "velocity", "forward"):
            va = _xyz(getattr(a.kinematic_state, name))
            vb = _xyz(_mirror(getattr(b.kinematic_state, name)))
            if not _close(va, vb, tol):
                out.append(f"{sid} {name}: {va} vs mirrored {_other(sid)} {vb}")
        scalars = (
            ("hull", a.hull_integrity, b.hull_integrity),
            ("propellant", a.kinematic_state.propellant_kg, b.kinematic_state.propellant_kg),
            ("throttle", getattr(a, "applied_throttle", 0.0), getattr(b, "applied_throttle", 0.0)),
            ("damage_taken", a.damage_taken_gj, b.damage_taken_gj),
            ("destroyed", float(a.is_destroyed), float(b.is_destroyed)),
            ("dying", float(a.is_dying), float(b.is_dying)),
        )
        for name, x, y in scalars:
            if not _close((x,), (y,), tol):
                out.append(f"{sid} {name}: {x} vs {_other(sid)} {y}")
        if a.execution_mode != b.execution_mode:
            out.append(f"{sid} mode {a.execution_mode} vs {_other(sid)} {b.execution_mode}")

    def flights(side):
        rows = [f for f in sim.torpedoes if f.source_ship_id.startswith(side)]
        return sorted(rows, key=lambda f: (f.source_ship_id.partition("_")[2],
                                           f.launch_time, f.torpedo_id[-4:]))
    ta, tb = flights("alpha"), flights("beta")
    if len(ta) != len(tb):
        out.append(f"torpedoes in flight: alpha {len(ta)} vs beta {len(tb)}")
    else:
        for fa, fb in zip(ta, tb):
            if not (_close(_xyz(fa.torpedo.position), _xyz(_mirror(fb.torpedo.position)), tol)
                    and _close((fa.heat_absorbed_j,), (fb.heat_absorbed_j,), tol)
                    and fa.is_disabled == fb.is_disabled):
                out.append(f"torpedo {fa.torpedo_id} vs {fb.torpedo_id}: "
                           f"{_xyz(fa.torpedo.position)} / {_xyz(_mirror(fb.torpedo.position))} "
                           f"heat {fa.heat_absorbed_j:.0f}/{fb.heat_absorbed_j:.0f} "
                           f"disabled {fa.is_disabled}/{fb.is_disabled}")
    na = sum(1 for f in sim.projectiles if f.source_ship_id.startswith("alpha"))
    nb = len(sim.projectiles) - na
    if na != nb:
        out.append(f"rounds in flight: alpha {na} vs beta {nb}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--max-ships", type=int, default=8)
    parser.add_argument("--doctrine", choices=DOCTRINES, default=DEFAULT_DOCTRINE)
    parser.add_argument("--distance", type=float, default=500.0)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--beta-first", action="store_true",
                        help="Add beta's ships to the simulation before alpha's")
    parser.add_argument("--flip", action="store_true",
                        help="Rotate the battlefield 180 degrees about Z before the first step")
    parser.add_argument("--flat-rng", type=float, nargs="?", const=0.31, default=None,
                        metavar="VALUE",
                        help="Make every random draw return VALUE (default 0.31; low "
                             "values make every hit, crit and fitting strike land)")
    parser.add_argument("--breaks", type=int, default=2,
                        help="Report the first N ticks that break the mirror (0 = result only)")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Relative tolerance of the mirror comparison")
    args = parser.parse_args()

    fleet_data = load_fleet_data()
    drafts = {f: auto_draft(f, budget=args.budget, max_ships=args.max_ships, seed=args.seed)
              for f in ("alpha", "beta")}
    # After the draft: the roster is still the seed's.
    real_random = random.Random
    if args.flat_rng is not None:
        FlatRandom.value = args.flat_rng
        random.Random = FlatRandom
    fleet_config = BattleFleetConfig(
        battle_name=f"Mirror probe ({args.budget} pts)",
        alpha_fleet=draft_to_fleet_definition(
            drafts["alpha"], args.distance, captain_model=HEURISTIC_MODEL, admiral_config=None),
        beta_fleet=draft_to_fleet_definition(
            drafts["beta"], args.distance, captain_model=HEURISTIC_MODEL, admiral_config=None),
        time_limit_s=args.time_limit, initial_distance_km=args.distance,
        record_battle=False, record_sim_trace=False,
        personality_selection=False, use_notebooks=False)
    battle_config = BattleConfig(
        initial_distance_km=args.distance, time_limit_s=args.time_limit,
        verbose=False, personality_selection=False,
        record_battle=False, record_sim_trace=False, record_predictions=False,
        seed=args.seed,
        heuristic_doctrine={"alpha": args.doctrine, "beta": args.doctrine})
    dummy = LLMCaptainConfig(name="-", ship_name="-", model=HEURISTIC_MODEL,
                             personality=CaptainPersonality.BALANCED)
    runner = LLMBattleRunner(config=battle_config, alpha_config=dummy, beta_config=dummy,
                             client=None, fleet_config=fleet_config)

    # Test-only hooks, applied to this process: the probes rearrange the
    # freshly set-up battle, and every step of the runner's simulation is
    # followed by the mirror check.
    setup = LLMBattleRunner.setup_fleet_battle

    def probed_setup(self, data):
        setup(self, data)
        sim = self.simulation
        sim.echo = False
        if args.flip:
            for ship in sim.ships.values():
                ks = ship.kinematic_state
                ks.position, ks.velocity, ks.forward = (
                    _mirror(ks.position), _mirror(ks.velocity), _mirror(ks.forward))
        if args.beta_first:
            ships = list(sim.ships.items())
            sim.ships.clear()
            sim.ships.update([s for s in ships if s[1].faction == "beta"]
                             + [s for s in ships if s[1].faction != "beta"])

    reported = [0]
    step = CombatSimulation.step

    def probed_step(self):
        events = step(self)
        if self is runner.simulation and reported[0] < args.breaks:
            breaks = mirror_breaks(self, args.tol)
            if breaks:
                reported[0] += 1
                kinds = Counter((e.event_type.name, (e.ship_id or "-").partition("_")[0])
                                for e in events)
                print(f"--- mirror broken after the step ending t={self.current_time:.1f} "
                      f"({len(breaks)} differences)")
                for line in breaks[:10]:
                    print("   ", line)
                print("    events this step:",
                      ", ".join(f"{k} {side} x{n}" for (k, side), n in sorted(kinds.items())))
                sys.stdout.flush()
        return events

    LLMBattleRunner.setup_fleet_battle = probed_setup
    CombatSimulation.step = probed_step
    try:
        result = runner.run_battle(fleet_data)
    finally:
        LLMBattleRunner.setup_fleet_battle = setup
        CombatSimulation.step = step
        random.Random = real_random

    sim = runner.simulation
    alive = {side: sum(1 for s in sim.ships.values()
                       if s.faction == side and not s.is_destroyed and not s.is_dying)
             for side in ("alpha", "beta")}
    roster = Counter(s.ship_type for s in sim.ships.values() if s.faction == "alpha")
    print(f"RESULT seed={args.seed} budget={args.budget} doctrine={args.doctrine} "
          f"beta_first={args.beta_first} flip={args.flip} flat_rng={args.flat_rng}: "
          f"winner={result.winner or 'draw'} ({result.reason}), {result.duration_s:.0f} s, "
          f"alive alpha {alive['alpha']} / beta {alive['beta']}; roster "
          + ", ".join(f"{n}x {t}" for t, n in sorted(roster.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
