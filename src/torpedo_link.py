"""
Torpedo datalink: one side's live-seeker rounds share a target board and
decide together who flies at which hull (pass 5).

Engine side, deterministic, no random draws, independent of ship and round
insertion order (everything is sorted by stable ids) and mirror-symmetric
(each side reads only its own rounds and the enemy hulls). It runs at the
start of CombatSimulation._update_torpedoes, so it reads the rounds at their
START-of-tick state and the ships at their END-of-tick state (the ship update
has already run) - the same snapshot the torpedo guidance that runs next
reads. CombatSimulation._torpedo_datalink applies the plan; this module holds
the pure parts:

- round_forecast() / need_for(): what a target needs from rounds LIKE THESE -
  point-defence soak plus rounds to kill, sized on each round's FORECAST
  impact (its closure now plus the delta-v it will still spend, the
  analytic twin of the captain's flown shot: within ~1 % of it from a
  launcher at rest or closing at 3 km/s) and its forecast crossing speed.
  kill_count() adds the measured LINK_KILL_MARGIN_ROUNDS. The ONE estimate:
  the datalink's board, the captain's OVERKILL line
  (src/llm/fire_solutions.py) and the admiral's TORPEDO ECONOMY
  (src/llm/fleet_intel.py, prompts.py) all call it, so they cannot disagree.
- reassign_gate(): can this round still steer onto that hull? An exact,
  cheap lower bound first (the coasting steer is the lateral relative
  speed; an opening target needs at least its opening speed), then the
  engine's can_intercept, then the burn time: the guidance burns at finite
  acceleration, so a steer of impulsive size dv_imp in t_go seconds costs
  dv = a t (1 - sqrt(1 - 2 dv_imp / (a t))) and is impossible above
  dv_imp = a t / 2. The reserve left after that cost grows with t_go.
- plan_group(): the group assignment: most constrained round first, each to
  the reachable target whose deficit - need minus the rounds already
  inbound that ARRIVE BEFORE it, minus every round assigned this tick - is still
  positive, best fuel (a PD-singed seeker: fastest) first; with no deficit
  anywhere, the least over-subscribed target.
"""

from __future__ import annotations

import math
import os
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


# -----------------------------------------------------------------------------
# Forecast impact and the shared need
# -----------------------------------------------------------------------------

# Delta-v a round still holds at impact on average (the NEZ reserve it keeps
# for late corrections): the forecast impact speed is closure + tank - this.
FORECAST_UNSPENT_KPS = 0.5
# Rounds the board keeps on a target beyond need_for().total before it calls
# the rest surplus, and the count the captain's OVERKILL line and the
# admiral's ENOUGH use. Measured (scripts/torpedo_link_margin.py): the
# smallest margin with which every flown strike at need + margin killed the
# target at least as often as the same strike with every round kept.
LINK_KILL_MARGIN_ROUNDS = 1


@dataclass
class Forecast:
    eta_s: float
    impact_kps: float
    cross_kps: float
    energy_gj: float


def round_forecast(torp: Any, target: Any) -> Forecast:
    """When and how hard this round arrives at ``target`` if it keeps flying
    at it: closure now plus the delta-v it will still spend
    (FORECAST_UNSPENT_KPS stays in the tank), crossing the PD envelope at its
    cruise floor (or faster if it already closes faster)."""
    try:
        from .torpedo import cruise_floor_kps
    except ImportError:  # pragma: no cover
        from torpedo import cruise_floor_kps
    rel = target.position - torp.position
    d = rel.magnitude
    closure = (-(target.velocity - torp.velocity).dot(rel / d) / 1000.0) if d > 0 else 0.0
    impact = max(0.0, closure) + max(0.0, torp.remaining_delta_v_kps - FORECAST_UNSPENT_KPS)
    cross = min(impact, max(closure, cruise_floor_kps(torp)))
    avg = 0.5 * (max(0.0, closure) + impact)
    pen = float(getattr(torp.specs, "penetrator_mass_kg", 250.0))
    warhead = float(getattr(torp.specs, "warhead_yield_gj", 0.0) or 0.0)
    return Forecast(eta_s=d / max(avg * 1000.0, 1.0), impact_kps=impact, cross_kps=cross,
                    energy_gj=0.5 * pen * (impact * 1000.0) ** 2 / 1e9 + warhead)


def need_for(sim: Any, target: Any, rounds: Sequence[Any],
             cache: Optional[Dict[Any, "RoundsNeeded"]] = None) -> "RoundsNeeded":
    """What ``target`` needs from rounds like ``rounds`` (TorpedoInFlight):
    rounds_needed() with their centroid as the threat direction and their
    mean forecast crossing speed and impact energy. ``cache`` is keyed by the
    target AND the exact round set it was computed from."""
    key = (target.ship_id, tuple(sorted(tf.torpedo_id for tf in rounds)))
    if cache is not None and key in cache:
        return cache[key]
    n = len(rounds)
    if n == 0:
        out = rounds_needed(sim, target)
    else:
        fc = [round_forecast(tf.torpedo, target) for tf in rounds]
        cx = sum(tf.torpedo.position.x for tf in rounds) / n
        cy = sum(tf.torpedo.position.y for tf in rounds) / n
        cz = sum(tf.torpedo.position.z for tf in rounds) / n
        centroid = type(rounds[0].torpedo.position)(cx, cy, cz)
        out = rounds_needed(sim, target, threat_from=centroid,
                            cross_kps=sum(f.cross_kps for f in fc) / n,
                            energy_gj=sum(f.energy_gj for f in fc) / n)
    if cache is not None:
        cache[key] = out
    return out


def kill_count(need: "RoundsNeeded") -> int:
    """Rounds the board keeps on a target (need + the measured margin): more
    inbound than this is surplus (OVERKILL for the commander)."""
    return need.total + LINK_KILL_MARGIN_ROUNDS


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
# Surplus diversion of reserve-profile rounds (measurement switch:
# AIC_LINK_SURPLUS=0 turns it off; group reassignment stays on).
LINK_SURPLUS_DIVERSION = os.environ.get("AIC_LINK_SURPLUS", "1") != "0"


def reassign_reserve_kps(t_go_s: float, cap_kps: float) -> float:
    return min(cap_kps, REASSIGN_RESERVE_BASE_KPS + REASSIGN_RESERVE_PER_S_KPS * max(0.0, t_go_s))


def burn_cost_kps(dv_imp_kps: float, accel_mps2: float, t_go_s: float) -> Optional[float]:
    """Delta-v a finite burn at ``accel_mps2`` spends to make the lateral
    displacement an impulsive steer of ``dv_imp_kps`` would make in
    ``t_go_s``: burn dv for dv/a seconds then coast covers
    dv (t - dv / 2a), so dv = a t (1 - sqrt(1 - 2 dv_imp / (a t))).
    None when impossible (dv_imp > a t / 2)."""
    if dv_imp_kps <= 0.0:
        return 0.0
    at = accel_mps2 * t_go_s / 1000.0
    if at <= 0.0 or dv_imp_kps > 0.5 * at:
        return None
    return at * (1.0 - math.sqrt(max(0.0, 1.0 - 2.0 * dv_imp_kps / at)))


def reassign_reach_km(torp: Any, to_go_km: float, closure_kps: float,
                      cap_kps: float = 1.12) -> float:
    """How far abeam (at the same range) a round ``to_go_km`` from its target,
    closing at ``closure_kps``, can be swung by a reassignment: the delta-v
    beyond the reassignment reserve, flown as a finite burn (burn_cost_kps)
    over the time to go."""
    if closure_kps <= 0:
        return 0.0
    t_go = to_go_km / closure_kps
    spare = torp.remaining_delta_v_kps - reassign_reserve_kps(t_go, cap_kps)
    if spare <= 0:
        return 0.0
    a = torp.specs.acceleration_at_mass(torp.current_mass_kg)
    at = a * t_go / 1000.0
    cost = min(spare, at)                      # cost a*t buys dv_imp = a*t/2
    dv_imp = cost * (1.0 - cost / (2.0 * at)) if at > 0 else 0.0
    return max(0.0, to_go_km * dv_imp / closure_kps)


@dataclass
class GateResult:
    ok: bool
    t_s: float
    fuel_left_kps: float
    reserve_kps: float
    gate: str           # ok | no_solution | dv | time | reserve


# Exact lower bounds may reject only when they exceed the tank by more than
# this relative slack (float noise between the closed form and can_intercept).
_PREFILTER_SLACK = 1e-9
# Measurement switch: AIC_LINK_PREFILTER=0 skips the lower-bound shortcut
# (the verdicts are identical either way; tests/test_torpedo_datalink.py).
PREFILTER = os.environ.get("AIC_LINK_PREFILTER", "1") != "0"


def _geometry(torp: Any, cand: Any) -> Tuple[float, float, float]:
    """(range m, along-LOS relative speed m/s (< 0: closing), lateral relative
    speed m/s) in plain floats - this runs for every orphan and enemy."""
    p, q = torp.position, cand.position
    rx, ry, rz = q.x - p.x, q.y - p.y, q.z - p.z
    d = math.sqrt(rx * rx + ry * ry + rz * rz)
    v, w = torp.velocity, cand.velocity
    ux, uy, uz = w.x - v.x, w.y - v.y, w.z - v.z
    if d <= 0.0:
        return 0.0, 0.0, 0.0
    lx, ly, lz = rx / d, ry / d, rz / d
    along = ux * lx + uy * ly + uz * lz
    ax, ay, az = ux - lx * along, uy - ly * along, uz - lz * along
    return d, along, math.sqrt(ax * ax + ay * ay + az * az)


def _cannot_reach(torp: Any, cand: Any) -> bool:
    """Exact lower bound on can_intercept's steer (no target acceleration):
    closing, the coasting intercept steers exactly the lateral relative speed
    (and a capped 3600 s solution steers more); opening, any intercept needs
    at least the opening speed."""
    d, along, lat = _geometry(torp, cand)
    if d <= 0.0:
        return False
    bound = (lat if along < 0.0 else along) / 1000.0
    return bound > torp.remaining_delta_v_kps * (1.0 + _PREFILTER_SLACK) + _PREFILTER_SLACK


def reassign_gate(torp: Any, cand: Any, max_t_s: float, cap_kps: float) -> GateResult:
    if PREFILTER and _cannot_reach(torp, cand):
        return GateResult(False, float("inf"), 0.0, 0.0, "dv")
    ok, t, left = torp.can_intercept(cand.position, cand.velocity)
    if math.isinf(t):
        return GateResult(False, t, 0.0, 0.0, "no_solution")
    if not ok:
        return GateResult(False, t, 0.0, 0.0, "dv")
    if t > max_t_s:
        return GateResult(False, t, left, 0.0, "time")
    if _geometry(torp, cand)[1] < 0.0:
        # Closing: can_intercept priced an instant sideways impulse at the
        # coasting time to go; the guidance burns it at finite acceleration.
        # (Not closing, can_intercept already solved a powered intercept at
        # the round's acceleration.)
        dv_imp = torp.remaining_delta_v_kps - left
        a = torp.specs.acceleration_at_mass(torp.current_mass_kg)
        cost = burn_cost_kps(dv_imp, a, t)
        if cost is None:
            return GateResult(False, t, 0.0, 0.0, "burn")
        left = torp.remaining_delta_v_kps - cost
        if left < 0.0:
            return GateResult(False, t, 0.0, 0.0, "dv")
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
               need: Dict[str, int],
               inbound_eta: Optional[Dict[str, List[float]]] = None,
               overflow: bool = True) -> Dict[str, str]:
    """Assign rounds (torpedo_id, reachable options) to targets.

    ``need``: rounds each target needs (kill_count); ``inbound_eta``: forecast
    arrival times of the rounds already homing on it. A round's deficit at a
    target is need - (inbound arriving before it) - (every round assigned to
    it this tick: they compete for the same need). Most constrained round first (fewest
    options, then id); each takes its best-scored option with a positive
    deficit; with none, the option with the largest deficit (overflow=False:
    stays). Ties go to the smaller ship id."""
    inbound_eta = inbound_eta or {}
    assigned: Dict[str, List[float]] = {}
    plan: Dict[str, str] = {}

    def deficit(o: Option) -> int:
        before = sum(1 for t in inbound_eta.get(o.target_id, ()) if t <= o.t_s)
        return need.get(o.target_id, 0) - before - len(assigned.get(o.target_id, ()))

    for tid, opts in sorted(rounds, key=lambda r: (len(r[1]), r[0])):
        if not opts:
            continue
        ordered = sorted(opts, key=lambda o: o.target_id)
        wanted = [o for o in ordered if deficit(o) > 0]
        if wanted:
            best = wanted[0]
            for o in wanted[1:]:
                if o.score > best.score:
                    best = o
        elif overflow:
            best = ordered[0]
            for o in ordered[1:]:
                if (deficit(o), o.score) > (deficit(best), best.score):
                    best = o
        else:
            continue
        plan[tid] = best.target_id
        assigned.setdefault(best.target_id, []).append(best.t_s)
    return plan


def plan_packets(rounds: Sequence[Tuple[str, List[Option]]], need: Dict[str, int],
                 inbound: Dict[str, int]) -> Dict[str, str]:
    """Surplus diversion as whole strikes: targets in order of how many of
    these rounds can reach them (then id); each takes its full deficit
    (need - inbound) from the rounds that reach it best (score, then id), or
    nothing when fewer than that can reach it - a partial packet would only
    trickle into its point defence."""
    opts = {tid: {o.target_id: o for o in o_list} for tid, o_list in rounds}
    reach: Dict[str, int] = {}
    for tid, by_t in opts.items():
        for sid in by_t:
            reach[sid] = reach.get(sid, 0) + 1
    free = set(opts)
    plan: Dict[str, str] = {}
    for sid in sorted(reach, key=lambda k: (-reach[k], k)):
        deficit = need.get(sid, 0) - inbound.get(sid, 0)
        if deficit <= 0:
            continue
        cands = sorted((tid for tid in free if sid in opts[tid]),
                       key=lambda tid: (-opts[tid][sid].score, tid))
        if len(cands) < deficit:
            continue
        for tid in cands[:deficit]:
            plan[tid] = sid
            free.discard(tid)
    return plan
