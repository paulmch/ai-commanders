"""
Torpedo datalink: one side's live-seeker rounds share a target board and
decide together who flies at which hull (pass 5).

Engine side, deterministic, no random draws, independent of ship and round
insertion order (everything is sorted by stable ids) and mirror-symmetric
(each side reads only its own rounds and the enemy hulls' start-of-tick
state). CombatSimulation._torpedo_datalink applies the plan; this module
holds the pure parts:

- rounds_needed(): what a target needs - point-defence soak plus rounds to
  kill. The ONE estimate shared with what commanders are shown (the
  captain's fire solution, src/llm/fire_solutions.py, and the admiral's
  torpedo economy, src/llm/fleet_intel.py / prompts.py, call it too), so the
  link and the display cannot disagree.
- reassign_gate(): can this round still steer onto that hull? The engine's
  own can_intercept (lateral steer at the coasting closure) plus a reserve
  that grows with the time to go (reassign_reserve_kps). A reassigned
  round's cruise floor is released (src/torpedo.py cruise_floor_kps), so the
  steer is all it has to pay for.
- plan_group(): the group assignment of a set of rounds onto targets with
  a remaining need: most constrained round first, each to the reachable
  target that still needs rounds and leaves it the most fuel (a PD-singed
  seeker: the fastest), the rest to the best reachable target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from . import weapon_arcs
    from .pointdefense import PDLaser, TORPEDO_ELECTRONICS_THRESHOLD_J
except ImportError:  # pragma: no cover - script-style imports
    import weapon_arcs
    from pointdefense import PDLaser, TORPEDO_ELECTRONICS_THRESHOLD_J


# =============================================================================
# What a target needs
# =============================================================================

# Rounds a target needs to die, from the measured salvo doctrine ("4 rounds
# kill a destroyer; 6 rounds kill anything up to a dreadnought" -
# ADMIRAL_DOCTRINE, re-measured by tests/test_doctrine_accuracy.py).
KILL_ROUNDS_LIGHT = 4
KILL_ROUNDS_HEAVY = 6
LIGHT_HULLS = frozenset({"corvette", "frigate", "destroyer"})
# The per-round impact energy those counts were measured at ("SIZING at
# moderate closure (~22 GJ/round ...)"). A slower round needs proportionally
# more company - a lower bound, since a weak round also penetrates worse.
KILL_ROUNDS_REF_GJ = 22.0
# Hull below this percentage counts as crippled: half the rounds (min 2).
KILL_ROUNDS_CRIPPLED_HULL_PCT = 50.0


def kill_rounds(ship_type: Optional[str], hull_pct: Optional[float] = None) -> int:
    """Simultaneous rounds that kill this hull (salvo doctrine), less if crippled."""
    base = KILL_ROUNDS_LIGHT if (ship_type or "") in LIGHT_HULLS else KILL_ROUNDS_HEAVY
    if hull_pct is not None and hull_pct < KILL_ROUNDS_CRIPPLED_HULL_PCT:
        base = max(2, base // 2)
    return base


def kill_rounds_at(rounds: int, energy_gj: Optional[float]) -> int:
    """``rounds`` (sized at KILL_ROUNDS_REF_GJ each) rescaled to rounds that land ``energy_gj``."""
    if energy_gj and 0.0 < energy_gj < KILL_ROUNDS_REF_GJ:
        return math.ceil(rounds * KILL_ROUNDS_REF_GJ / energy_gj)
    return rounds


def _laser(fleet_data: Optional[Dict[str, Any]] = None) -> PDLaser:
    spec = ((fleet_data or {}).get("weapon_types") or {}).get("pd_laser") or {}
    return PDLaser.from_fleet_data(spec) if spec else PDLaser()


def turrets_to_blind(cross_kps: float, fleet_data: Optional[Dict[str, Any]] = None,
                     laser: Optional[PDLaser] = None) -> int:
    """PD turrets that must dwell on ONE round crossing the envelope at
    ``cross_kps`` to seeker-kill it (the continuous-dwell closed form,
    PDLaser.energy_before_impact_j)."""
    laser = laser or _laser(fleet_data)
    per_turret = laser.energy_before_impact_j(laser.range_km, max(1.0, cross_kps) * 1000.0)
    if per_turret <= 0:
        return 99
    return max(1, math.ceil(TORPEDO_ELECTRONICS_THRESHOLD_J / per_turret))


def _arcs(sim: Any, ship: Any):
    if not getattr(sim, "weapon_arcs", False) or not hasattr(sim, "_arcs_for"):
        return None
    try:
        arcs = sim._arcs_for(ship)
        ks = ship.kinematic_state
        arcs.set_frame(ks.forward, ks.up, getattr(sim, "current_time", 0.0))
    except Exception:  # noqa: BLE001 - advisory callers pass test doubles
        return None
    return arcs if arcs.mounts else None


def pd_turrets_bearing(sim: Any, ship: Any, point: Any) -> Tuple[int, int]:
    """(bearing, operational): the ship's live PD turrets, and how many of them
    have a line of sight to ``point`` that their own hull leaves open. Without
    weapon arcs (or for a hull without mounts) every live turret bears."""
    turrets = [pd for pd in (getattr(ship, "point_defense", None) or [])
               if getattr(pd, "is_operational", True)]
    arcs = _arcs(sim, ship) if turrets and point is not None else None
    if arcs is None:
        return len(turrets), len(turrets)
    dx, dy, dz = (point.x - ship.position.x, point.y - ship.position.y,
                  point.z - ship.position.z)
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n <= 0.0:
        return len(turrets), len(turrets)
    cell = weapon_arcs.cell_index(*arcs.to_body(dx / n, dy / n, dz / n))
    bearing = sum(1 for pd in turrets
                  if getattr(pd, "mount", None) is None or not pd.mount.mask[cell])
    return bearing, len(turrets)


def _in_fight(ship: Any) -> bool:
    return (ship is not None and not getattr(ship, "is_destroyed", False)
            and not getattr(ship, "is_dying", False)
            and not getattr(ship, "is_surrendered", False))


def pd_cover(sim: Any, target: Any, threat_from: Any = None,
             radius_m: Optional[float] = None) -> Tuple[int, List[Tuple[Any, int]]]:
    """(own, [(escort ship, turrets)]): the target's live PD turrets and its
    in-fight allies' within ``radius_m`` (default: the PD envelope) - only
    those that BEAR on ``threat_from`` when it is given and arcs are on."""
    if radius_m is None:
        radius_m = _laser().range_km * 1000.0

    def live(ship) -> int:
        if threat_from is not None:
            return pd_turrets_bearing(sim, ship, threat_from)[0]
        return sum(1 for pd in (getattr(ship, "point_defense", None) or [])
                   if getattr(pd, "is_operational", True))

    own = live(target)
    escorts = []
    for ship in sorted(getattr(sim, "ships", {}).values(), key=lambda s: s.ship_id):
        if (ship is target or ship.faction != target.faction or not _in_fight(ship)):
            continue
        if (ship.position - target.position).magnitude > radius_m:
            continue
        n = live(ship)
        if n:
            escorts.append((ship, n))
    return own, escorts


@dataclass
class RoundsNeeded:
    """What one target needs: rounds the point defence soaks (blinds) out of a
    salvo arriving together, plus rounds that must land to kill it."""
    cover: int          # PD turrets that bear (own + escorts in the envelope)
    soak: int           # rounds blinded out of a salvo arriving together
    kill: int           # rounds that must land (at the given energy)
    @property
    def total(self) -> int:
        return self.soak + self.kill


def rounds_needed(sim: Any, target: Any, threat_from: Any = None,
                  cross_kps: Optional[float] = None, energy_gj: Optional[float] = None,
                  fleet_data: Optional[Dict[str, Any]] = None,
                  cover: Optional[int] = None) -> RoundsNeeded:
    """The shared estimate (see the module docstring). ``cross_kps``: closure
    at which the rounds cross the PD envelope (default: the sprint cruise
    floor); ``energy_gj``: per-round impact energy (default: the reference);
    ``cover``: a turret count the caller already has (else counted here)."""
    try:
        from .torpedo import MIN_CLOSING_SPEED_KPS
    except ImportError:  # pragma: no cover - script-style imports
        from torpedo import MIN_CLOSING_SPEED_KPS
    if cover is None:
        own, escorts = pd_cover(sim, target, threat_from)
        cover = own + sum(n for _, n in escorts)
    laser = None
    if fleet_data is None:
        # the engine's own PD lasers (the same fleet-data spec on every hull)
        for pd in (getattr(target, "point_defense", None) or []):
            laser = getattr(pd, "laser", None)
            if laser is not None:
                break
    per = turrets_to_blind(MIN_CLOSING_SPEED_KPS if cross_kps is None else cross_kps,
                           fleet_data, laser=laser)
    soak = cover // max(1, per)
    kill = kill_rounds_at(kill_rounds(getattr(target, "ship_type", None),
                                      getattr(target, "hull_integrity", None)), energy_gj)
    return RoundsNeeded(cover=cover, soak=soak, kill=kill)


# =============================================================================
# Reassignment gate
# =============================================================================

# Delta-v a reassigned round must still hold AFTER the steer, as a function
# of the time to go: RESERVE_BASE + RESERVE_PER_S * t, capped at the old
# fixed reserve (CombatSimulation.RETARGET_MIN_RESERVE_KPS / 8 % of the tank).
# Re-derived from flown reassignments (scripts/torpedo_link_gate.py): the
# reserve pays for the target's own motion during the chase, which grows with
# the time to go; a short chase needs little. See the pass-5 report.
REASSIGN_RESERVE_BASE_KPS = 0.25
REASSIGN_RESERVE_PER_S_KPS = 0.02
# A round this close to impact on its target is committed: the link never
# diverts it (it may still be reassigned if its target dies).
LINK_COMMITTED_S = 12.0
# Surplus diversion looks again this often (s): the board is stable between
# kills, and a round must not be swung back and forth.
LINK_SURPLUS_INTERVAL_S = 5.0


def reassign_reserve_kps(t_go_s: float, cap_kps: float) -> float:
    return min(cap_kps, REASSIGN_RESERVE_BASE_KPS + REASSIGN_RESERVE_PER_S_KPS * max(0.0, t_go_s))


@dataclass
class GateResult:
    ok: bool
    t_s: float
    fuel_left_kps: float
    reserve_kps: float
    gate: str           # ok | no_solution | dv | time | reserve


def reassign_gate(torp: Any, cand: Any, max_t_s: float, cap_kps: float) -> GateResult:
    ok, t, left = torp.can_intercept(cand.position, cand.velocity)
    if math.isinf(t):
        return GateResult(False, t, 0.0, 0.0, "no_solution")
    if not ok:
        return GateResult(False, t, 0.0, 0.0, "dv")
    if t > max_t_s:
        return GateResult(False, t, left, 0.0, "time")
    reserve = reassign_reserve_kps(t, cap_kps)
    if left < reserve:
        return GateResult(False, t, left, reserve, "reserve")
    return GateResult(True, t, left, reserve, "ok")


# =============================================================================
# Group planning
# =============================================================================

@dataclass
class Option:
    target_id: str
    score: float
    t_s: float
    fuel_left_kps: float


def plan_group(rounds: Sequence[Tuple[str, List[Option]]],
               remaining_need: Dict[str, int], overflow: bool = True) -> Dict[str, str]:
    """Assign rounds (torpedo_id, reachable options) to targets.

    Most constrained round first (fewest options, then id); each takes the
    best-scored option among targets that still need rounds, else the
    best-scored reachable one (``overflow=False``: no option without need
    is taken - the round stays). Ties go to the smaller ship id. Returns
    {torpedo_id: target_id} for every round with at least one option;
    ``remaining_need`` is decremented in place."""
    plan: Dict[str, str] = {}
    for tid, opts in sorted(rounds, key=lambda r: (len(r[1]), r[0])):
        if not opts:
            continue
        ordered = sorted(opts, key=lambda o: o.target_id)
        wanted = [o for o in ordered if remaining_need.get(o.target_id, 0) > 0]
        pool = wanted or (ordered if overflow else [])
        if not pool:
            continue
        best = pool[0]
        for o in pool[1:]:
            if o.score > best.score:
                best = o
        plan[tid] = best.target_id
        remaining_need[best.target_id] = remaining_need.get(best.target_id, 0) - 1
    return plan
