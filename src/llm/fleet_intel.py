"""
Fleet intelligence: the derived battle facts a commander decides on.

Raw simulation state says where everything is; a commander needs what it
MEANS - which rounds in flight can still hurt whom, whether a torpedo launched
now can catch a target that runs, how many rounds an escort wall soaks up,
what changed since the last decision and who did it. The admiral's turn state
(src/llm/prompts.py), its per-ship order prompts and its tactical plot
(src/llm/admiral_view.py) are all built on this module.

Every function is a pure read of the simulation: nothing here mutates the
battle or draws from its RNG, so seeded battles are unchanged by it. Where
the engine already answers the same question, its rule is reused rather than
re-invented - a seeker-killed torpedo threatens a hull by exactly the
closest-approach test the engine's own evasion uses
(CombatSimulation._gather_ballistic_threats / _calculate_evasion).

Audit references (scratchpad findings): ADM-01 (projectiles), ADM-05/R5
(torpedo solutions, salvo sizing vs PD), ADM-06 (since-last-checkpoint
digest), ADM-13 (one label format for every ship).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from ..physics import G_STANDARD, Vector3D
    from ..pointdefense import PDLaser, TORPEDO_ELECTRONICS_THRESHOLD_J
    from ..torpedo import MIN_CLOSING_SPEED_KPS, TorpedoGuidance
except ImportError:  # pragma: no cover - script-style imports
    from physics import G_STANDARD, Vector3D
    from pointdefense import PDLaser, TORPEDO_ELECTRONICS_THRESHOLD_J
    from torpedo import MIN_CLOSING_SPEED_KPS, TorpedoGuidance

from .battle_ticker import TYPE_ABBR


# A ballistic mass (slug, seeker-killed or dry torpedo) threatens a hull when
# its straight-line closest approach falls inside this miss distance within
# the engine's BALLISTIC_THREAT_HORIZON_S. Same numbers as the engine's evasion trigger:
# _calculate_evasion treats a ballistic threat as dangerous at miss <=
# 10 x ship_radius_m (50 m), and _gather_ballistic_threats only looks
# BALLISTIC_THREAT_HORIZON_S ahead.
WRECK_THREAT_MISS_M = 10 * 50.0

def ballistic_horizon_s() -> float:
    """The engine's BALLISTIC_THREAT_HORIZON_S (imported lazily: the simulation
    module only imports as part of the src package)."""
    try:
        from ..simulation import BALLISTIC_THREAT_HORIZON_S
    except ImportError:  # pragma: no cover - script-style imports
        from simulation import BALLISTIC_THREAT_HORIZON_S
    return float(BALLISTIC_THREAT_HORIZON_S)


# Relative speeds below this are "parked" - the engine ignores them too.
BALLISTIC_MIN_REL_SPEED_MPS = 100.0

# Torpedo solution model: how far ahead to fly the 1-D pursuit and its step.
SOLUTION_HORIZON_S = 900.0
SOLUTION_STEP_S = 0.5
# The terminal burn only spends the round's NEZ surplus in its last moments:
# model it as at most this many seconds of full thrust at the end. Fitted to
# the measured doctrine anchors (zero-closure launch -> ~13.5 km/s, closing
# 12 km/s -> 14-17 km/s, head-on 26 km/s -> ~28 km/s; tests pin all three).
TERMINAL_WINDOW_S = 25.0
# Verdict thresholds on the impact energy of a round whose target RUNS. A
# zero-closure launch lands ~13.5 km/s = ~23 GJ (doctrine anchor); a round
# lobbed at a receding ship ~8 GJ "if it lands at all". GOOD needs at least
# GOOD_ENERGY_GJ (~3.5 spinal hits); below NO_ENERGY_GJ (about one spinal
# hit) the round is not worth its magazine slot.
GOOD_ENERGY_GJ = 15.0
NO_ENERGY_GJ = 5.0
# Opening faster than this counts as receding (the doctrine's "never launch
# at a receding evading ship"): such a solution is never better than POOR.
RECEDING_KPS = 0.5

# Rounds a target needs to die, from the measured salvo doctrine ("4 rounds
# kill a destroyer; 6 rounds kill anything up to a dreadnought" -
# ADMIRAL_DOCTRINE, re-measured by tests/test_doctrine_accuracy.py).
KILL_ROUNDS_LIGHT = 4
KILL_ROUNDS_HEAVY = 6
LIGHT_HULLS = frozenset({"corvette", "frigate", "destroyer"})


# =============================================================================
# Labels
# =============================================================================

def ship_name(ship: Any) -> str:
    return getattr(ship, "name", None) or getattr(ship, "ship_id", "?")


def label(name: Optional[str], ship_id: Optional[str]) -> str:
    """'TIS Dart-7', 'alpha_7' -> 'TIS Dart-7 [alpha_7]' (the one format everywhere).

    Full name AND id: mirrored drafts can field same-named hulls on both
    sides ('Dart-1'), and engine feedback speaks in ids - the pair resolves both.
    """
    if not ship_id:
        return name or "?"
    if not name or name == ship_id:
        return f"[{ship_id}]"
    return f"{name} [{ship_id}]"


def ship_label(ship: Any) -> str:
    return label(ship_name(ship), getattr(ship, "ship_id", None))


def class_abbr(ship_type: Optional[str]) -> str:
    return TYPE_ABBR.get(ship_type or "", str(ship_type or "?")[:2].upper())


def is_live(ship: Any) -> bool:
    """In the fight: not destroyed, not a dying hulk, not surrendered."""
    return (ship is not None and not getattr(ship, "is_destroyed", False)
            and not getattr(ship, "is_dying", False)
            and not getattr(ship, "is_surrendered", False))


def faction_of(sim: Any, ship_id: Optional[str]) -> Optional[str]:
    ship = sim.get_ship(ship_id) if ship_id else None
    if ship is not None:
        return getattr(ship, "faction", None)
    if ship_id and "_" in ship_id:
        return ship_id.split("_", 1)[0]
    return None


def max_accel_ms2(ship: Any) -> float:
    """Full-throttle acceleration available now (engine damage included)."""
    try:
        ks = ship.kinematic_state
        eff = ship.get_effective_thrust_fraction() if hasattr(ship, "get_effective_thrust_fraction") else 1.0
        return ks.max_acceleration_ms2() * eff
    except AttributeError:
        return 0.0


def pd_turrets(ship: Any) -> Tuple[int, int]:
    """(operational, total) point-defense turrets."""
    pds = getattr(ship, "point_defense", None) or []
    return sum(1 for pd in pds if getattr(pd, "is_operational", True)), len(pds)


def launch_capacity(sim: Any, ship: Any) -> int:
    """Torpedoes this ship can still put out before the next checkpoint."""
    if not getattr(ship, "torpedoes_remaining", 0):
        return 0
    capacity = getattr(sim, "torpedo_launch_capacity", None)
    if capacity is None:
        return 0
    try:
        return int(capacity(ship.ship_id))
    except Exception:  # noqa: BLE001 - advisory number only
        return 0


# =============================================================================
# Ballistic closest approach (engine rule)
# =============================================================================

def closest_approach(rel_pos: Vector3D, rel_vel: Vector3D) -> Tuple[float, float]:
    """(seconds to closest approach, miss distance m) of a straight-line pass."""
    vv = rel_vel.dot(rel_vel)
    if vv < 1e-9:
        return math.inf, rel_pos.magnitude
    tca = -rel_pos.dot(rel_vel) / vv
    return tca, (rel_pos + rel_vel * tca).magnitude


def ballistic_threat(pos: Vector3D, vel: Vector3D, ships: Iterable[Any],
                     horizon_s: Optional[float] = None,
                     miss_m: float = WRECK_THREAT_MISS_M) -> Optional[Tuple[Any, float, float]]:
    """
    The hull a coasting mass will strike (earliest), as (ship, tca_s, miss_m).

    The engine's evasion test, applied to every candidate: closest approach
    ahead within ``horizon_s`` and inside ``miss_m``. A receding wreck (tca
    in the past) or one passing wide is no threat to anyone.
    """
    if horizon_s is None:
        horizon_s = ballistic_horizon_s()
    best = None
    for ship in ships:
        rel_pos = pos - ship.position
        rel_vel = vel - ship.velocity
        if rel_vel.magnitude < BALLISTIC_MIN_REL_SPEED_MPS:
            continue
        tca, miss = closest_approach(rel_pos, rel_vel)
        if tca < 0 or tca > horizon_s or miss > miss_m:
            continue
        if best is None or tca < best[1]:
            best = (ship, tca, miss)
    return best


def _closing(rel_pos: Vector3D, rel_vel: Vector3D) -> float:
    """Closing speed (m/s, + closing) of something at rel_pos moving at rel_vel."""
    d = rel_pos.magnitude
    return -rel_pos.dot(rel_vel) / d if d > 0 else 0.0


# =============================================================================
# Torpedo solution (1-D pursuit model)
# =============================================================================

@dataclass
class TorpedoSolution:
    """Can a torpedo launched now (or already flying) catch a target that runs?"""
    verdict: str                      # GOOD | POOR | NO
    range_km: float
    closing_kps: float
    tof_s: Optional[float] = None     # time of flight if he runs
    cross_kps: Optional[float] = None  # mean closure inside the PD envelope
    impact_kps: Optional[float] = None
    energy_gj: Optional[float] = None
    short_km: Optional[float] = None  # gap left when it runs dry (NO)
    pd_cover: Optional[int] = None    # operational PD turrets within reach of the target
    soak: Optional[int] = None        # rounds that cover blinds at the crossing speed
    reason: str = ""

    @property
    def reaches(self) -> bool:
        return self.verdict in ("GOOD", "POOR")


def torpedo_specs_for(sim: Any, fleet_data: Optional[Dict[str, Any]],
                      ships: Sequence[Any] = ()) -> Any:
    """The torpedo specs the engine flies: from a live launcher, else fleet data."""
    for ship in ships:
        for launcher in getattr(ship, "ready_torpedo_launchers", None) or []:
            specs = getattr(launcher, "specs", None)
            if specs is not None:
                return specs
    try:
        from ..simulation import _build_torpedo_specs
    except ImportError:  # pragma: no cover
        from simulation import _build_torpedo_specs
    data = ((fleet_data or {}).get("weapon_types") or {}).get("torpedo_launcher") or {}
    return _build_torpedo_specs(data)


def turrets_to_blind(fleet_data: Optional[Dict[str, Any]], cross_kps: float) -> int:
    """
    PD turrets that must dwell on ONE round crossing the envelope at
    ``cross_kps`` to seeker-kill it (same closed form the PD doctrine quotes:
    pd_doctrine_numbers / PDLaser.energy_before_impact_j).
    """
    spec = ((fleet_data or {}).get("weapon_types") or {}).get("pd_laser") or {}
    laser = PDLaser.from_fleet_data(spec) if spec else PDLaser()
    per_turret = laser.energy_before_impact_j(laser.range_km, max(1.0, cross_kps) * 1000.0)
    if per_turret <= 0:
        return 99
    return max(1, math.ceil(TORPEDO_ELECTRONICS_THRESHOLD_J / per_turret))


def pd_range_m(fleet_data: Optional[Dict[str, Any]]) -> float:
    spec = ((fleet_data or {}).get("weapon_types") or {}).get("pd_laser") or {}
    return float(spec.get("range_km", 250.0)) * 1000.0


def pd_cover(sim: Any, target: Any, radius_m: float) -> int:
    """Operational PD turrets of the target's side within ``radius_m`` of it."""
    total = 0
    for ship in getattr(sim, "ships", {}).values():
        if ship.faction != target.faction or not is_live(ship):
            continue
        if (ship.position - target.position).magnitude <= radius_m:
            total += pd_turrets(ship)[0]
    return total


def solve_pursuit(
    range_m: float,
    closing_mps: float,
    lateral_mps: float,
    target_accel_ms2: float,
    specs: Any,
    mass_kg: Optional[float] = None,
    dv_left_mps: Optional[float] = None,
    pd_envelope_m: float = 250_000.0,
    floor_mps: float = MIN_CLOSING_SPEED_KPS * 1000.0,
    horizon_s: float = SOLUTION_HORIZON_S,
) -> Dict[str, Any]:
    """
    Fly a 1-D pursuit along the line of sight, worst case for the torpedo:
    the target RUNS (full acceleration straight away) for the whole flight.

    The torpedo first spends whatever nulls the crossing (lateral) speed, then
    flies its guidance law: burn only while closure is under the floor
    (MIN_CLOSING_SPEED_KPS), then a terminal burn of what is left (at most
    TERMINAL_WINDOW_S of thrust). Thrust is
    the specs' thrust over the falling mass (rocket equation), so the burn
    shortens as propellant goes - the same drive the engine integrates.

    Returns hit (bool), tof_s, impact_mps, cross_mps (mean closure while
    inside the PD envelope - what PD dwell depends on), short_m.
    """
    ve = max(1.0, float(specs.exhaust_velocity_kps) * 1000.0)
    m = float(mass_kg if mass_kg is not None else specs.mass_kg)
    dv = (float(dv_left_mps) if dv_left_mps is not None
          else ve * math.log(max(1.0, m / max(1.0, specs.dry_mass_kg))))
    dv -= max(0.0, lateral_mps)
    if dv < 0:
        return {"hit": False, "short_m": range_m, "reason": "crossing too fast"}
    c, d, t = closing_mps, range_m, 0.0
    inside_t = 0.0
    inside_start_d = min(d, pd_envelope_m)
    dt = SOLUTION_STEP_S
    while t < horizon_s:
        if c < floor_mps and dv > 0:
            a = specs.thrust_n / max(1.0, m)
            burn = min(a * dt, floor_mps - c, dv)
            c += burn
            dv -= burn
            m *= math.exp(-burn / ve)
        c -= target_accel_ms2 * dt
        d -= c * dt
        t += dt
        if d <= pd_envelope_m:
            inside_t += dt
        if d <= 0:
            cross = inside_start_d / inside_t if inside_t > 0 else c
            dump = min(max(0.0, dv), specs.thrust_n / max(1.0, m) * TERMINAL_WINDOW_S)
            return {"hit": True, "tof_s": t, "impact_mps": c + dump, "cross_mps": cross}
        if c <= 0 and dv <= 0:
            return {"hit": False, "short_m": d, "reason": "runs dry"}
    return {"hit": False, "short_m": d, "reason": "too slow"}


def torpedo_solution(
    launch_pos: Vector3D,
    launch_vel: Vector3D,
    target: Any,
    specs: Any,
    fleet_data: Optional[Dict[str, Any]] = None,
    sim: Any = None,
    salvo_available: Optional[int] = None,
    mass_kg: Optional[float] = None,
    dv_left_mps: Optional[float] = None,
) -> TorpedoSolution:
    """
    Verdict for a round launched from (launch_pos, launch_vel) at ``target``
    (or, with mass_kg/dv_left_mps, for a round already in flight there),
    judged on the worst case - the target RUNS for the whole flight, which is
    what an EVADE-ordered hull does against a guided torpedo:

    GOOD - not receding, reaches, and lands >= GOOD_ENERGY_GJ.
    POOR - reaches, but receding or lands < GOOD_ENERGY_GJ: slow crossing,
           long PD dwell, a fraction of a closing shot's damage.
    NO   - beyond launcher reach; a running target outlasts the round (it
           runs dry short); it lands < NO_ENERGY_GJ; or, receding, the PD
           around the target blinds more slow rounds than the salvo you have.

    Measured on battle A: launches closing 2-4 km/s connected (16 hits);
    all 8 rounds launched at T+270 into a 2 g destroyer opening 3-5 km/s
    burned dry or were blinded (0 hits).
    """
    rel_pos = target.position - launch_pos
    rel_vel = target.velocity - launch_vel
    rng = rel_pos.magnitude
    closing = _closing(rel_pos, rel_vel)
    los = rel_pos.normalized() if rng > 1.0 else Vector3D(1.0, 0.0, 0.0)
    lateral = (rel_vel - los * rel_vel.dot(los)).magnitude
    sol = TorpedoSolution(verdict="NO", range_km=rng / 1000.0, closing_kps=closing / 1000.0)
    max_range_m = float(((fleet_data or {}).get("weapon_types") or {})
                        .get("torpedo_launcher", {}).get("range_km", 2500.0)) * 1000.0
    if mass_kg is None and rng > max_range_m:
        sol.reason = f"beyond {max_range_m / 1000.0:.0f} km reach"
        return sol
    envelope = pd_range_m(fleet_data)
    run = solve_pursuit(rng, closing, lateral, max_accel_ms2(target), specs,
                        mass_kg=mass_kg, dv_left_mps=dv_left_mps, pd_envelope_m=envelope)
    if not run["hit"]:
        sol.short_km = run.get("short_m", 0.0) / 1000.0
        sol.reason = ("runs dry ~%.0f km short if he runs" % sol.short_km
                      if run.get("reason") == "runs dry" else
                      "cannot null his crossing speed" if run.get("reason") == "crossing too fast"
                      else "too slow to catch a runner")
        return sol
    sol.tof_s = run["tof_s"]
    sol.impact_kps = run["impact_mps"] / 1000.0
    sol.cross_kps = run["cross_mps"] / 1000.0
    pen = float(getattr(specs, "penetrator_mass_kg", 250.0))
    sol.energy_gj = (0.5 * pen * run["impact_mps"] ** 2 / 1e9
                     + float(getattr(specs, "warhead_yield_gj", 0.0) or 0.0))
    if sim is not None:
        sol.pd_cover = pd_cover(sim, target, envelope)
        sol.soak = sol.pd_cover // turrets_to_blind(fleet_data, sol.cross_kps)
    receding = sol.closing_kps < -RECEDING_KPS
    if sol.energy_gj < NO_ENERGY_GJ:
        sol.reason = f"lands only ~{sol.energy_gj:.0f} GJ if he runs"
        return sol
    if (receding and sol.soak is not None and salvo_available is not None
            and sol.soak >= max(1, salvo_available)):
        sol.reason = (f"receding: crosses PD at {sol.cross_kps:.1f} km/s, his {sol.pd_cover} "
                      f"turrets blind {sol.soak} slow rounds >= your {salvo_available}")
        return sol
    if receding:
        sol.verdict = "POOR"
        sol.reason = f"receding {-sol.closing_kps:.1f} km/s"
    elif sol.energy_gj < GOOD_ENERGY_GJ:
        sol.verdict = "POOR"
        sol.reason = "slow arrival"
    else:
        sol.verdict = "GOOD"
    return sol


def kill_rounds(ship_type: Optional[str], hull_pct: Optional[float] = None) -> int:
    """Simultaneous rounds that kill this hull (salvo doctrine), less if crippled."""
    base = KILL_ROUNDS_LIGHT if (ship_type or "") in LIGHT_HULLS else KILL_ROUNDS_HEAVY
    if hull_pct is not None and hull_pct < 50.0:
        base = max(2, base // 2)
    return base


# =============================================================================
# Ordnance in flight
# =============================================================================

@dataclass
class RoundInfo:
    """One round in flight, attributed to a side and classified by state."""
    kind: str                         # torpedo | slug
    side: str                         # ours | theirs
    state: str                        # guided | blinded | dry | orphaned | ballistic
    round_id: str
    source_id: Optional[str]
    target_id: Optional[str]
    weapon: str
    range_km: Optional[float] = None  # to its target (guided) / threatened hull (ballistic)
    closing_kps: Optional[float] = None
    eta_s: Optional[float] = None     # None: not closing
    energy_gj: float = 0.0
    threat_id: Optional[str] = None   # hull it will strike (ballistic) / is homing on (guided)
    tca_s: Optional[float] = None
    miss_km: Optional[float] = None
    nez_inside: Optional[bool] = None
    dv_left_kps: Optional[float] = None
    can_reach: Optional[bool] = None  # our guided rounds: still catches a runner

    @property
    def ballistic(self) -> bool:
        return self.state in ("blinded", "dry", "orphaned", "ballistic")


def gather_rounds(sim: Any, faction: str, specs: Any = None) -> List[RoundInfo]:
    """
    Every round in flight, side-attributed and classified (ADM-01).

    Guided torpedoes are measured against their target; ballistic masses
    (blinded, dry or orphaned torpedoes) against every live hull with the
    engine's closest-approach test - a wreck on nobody's line is listed as
    such, never as "inbound".
    """
    live = [s for s in getattr(sim, "ships", {}).values() if is_live(s)]
    out: List[RoundInfo] = []
    guidance = TorpedoGuidance()
    for tf in getattr(sim, "torpedoes", None) or []:
        torp = tf.torpedo
        side = "ours" if faction_of(sim, tf.source_ship_id) == faction else "theirs"
        target = sim.get_ship(torp.target_id)
        target_alive = target is not None and not target.is_destroyed
        if tf.is_disabled:
            state = "blinded"
        elif not target_alive:
            state = "orphaned"
        elif getattr(torp, "fuel_exhausted", False):
            state = "dry"
        else:
            state = "guided"
        info = RoundInfo(kind="torpedo", side=side, state=state, round_id=tf.torpedo_id,
                         source_id=tf.source_ship_id, target_id=torp.target_id,
                         weapon="Torpedo",
                         dv_left_kps=float(getattr(torp, "remaining_delta_v_kps", 0.0) or 0.0))
        if state == "guided":
            rel_pos = torp.position - target.position
            rel_vel = torp.velocity - target.velocity
            dist = rel_pos.magnitude
            closing = _closing(rel_pos, rel_vel)
            eta = dist / closing if closing > 1.0 else None
            info.range_km, info.closing_kps, info.eta_s = dist / 1000.0, closing / 1000.0, eta
            # Projected impact: current relative speed plus the closure the
            # remaining burn can still add (the engine's threat estimate).
            if eta is not None:
                gain = min(info.dv_left_kps * 1000.0,
                           torp.specs.acceleration_at_mass(torp.current_mass_kg) * eta)
            else:
                gain = info.dv_left_kps * 1000.0
            speed = rel_vel.magnitude + max(0.0, gain)
            info.energy_gj = (0.5 * torp.specs.penetrator_mass_kg * speed ** 2 / 1e9
                              + torp.specs.warhead_yield_gj)
            info.threat_id = target.ship_id
            if side == "theirs" and target.faction == faction:
                try:
                    info.nez_inside = bool(guidance.nez_status(
                        torp, target.position, target.velocity,
                        target_max_lateral_accel_ms2=max_accel_ms2(target)).inside)
                except Exception:  # noqa: BLE001 - advisory only
                    info.nez_inside = None
            if side == "ours" and specs is not None:
                sol = torpedo_solution(torp.position, torp.velocity, target, specs,
                                       mass_kg=torp.current_mass_kg,
                                       dv_left_mps=info.dv_left_kps * 1000.0)
                info.can_reach = sol.reaches
        else:
            hit = ballistic_threat(torp.position, torp.velocity, live)
            if hit is not None:
                ship, tca, miss = hit
                rel_vel = torp.velocity - ship.velocity
                info.threat_id = ship.ship_id
                info.tca_s, info.miss_km = tca, miss / 1000.0
                info.range_km = (torp.position - ship.position).magnitude / 1000.0
                info.closing_kps = _closing(torp.position - ship.position, rel_vel) / 1000.0
                info.eta_s = tca
                # A dead seeker cannot burn: the impact is the relative speed now.
                info.energy_gj = 0.5 * torp.specs.penetrator_mass_kg * rel_vel.magnitude ** 2 / 1e9
        out.append(info)

    for pf in getattr(sim, "projectiles", None) or []:
        proj = pf.projectile
        side = "ours" if faction_of(sim, pf.source_ship_id) == faction else "theirs"
        target = sim.get_ship(getattr(pf, "target_ship_id", None))
        info = RoundInfo(kind="slug", side=side, state="ballistic", round_id=pf.projectile_id,
                         source_id=pf.source_ship_id, target_id=getattr(pf, "target_ship_id", None),
                         weapon=_weapon_label(pf))
        if target is not None and not target.is_destroyed:
            rel_pos = proj.position - target.position
            rel_vel = proj.velocity - target.velocity
            closing = _closing(rel_pos, rel_vel)
            info.range_km = rel_pos.magnitude / 1000.0
            info.closing_kps = closing / 1000.0
            info.eta_s = rel_pos.magnitude / closing if closing > 1.0 else None
            mass = max(0.1, proj.mass_kg - getattr(proj, "_pd_ablation", 0.0))
            info.energy_gj = 0.5 * mass * rel_vel.magnitude ** 2 / 1e9
            if info.eta_s is not None:
                info.threat_id = target.ship_id
        out.append(info)
    return out


def _weapon_label(pf: Any) -> str:
    weapon = getattr(pf, "weapon", None)
    name = getattr(weapon, "name", None) if weapon is not None else None
    if name:
        return str(name)
    wtype = getattr(weapon, "weapon_type", None) if weapon is not None else None
    if wtype:
        return str(wtype)
    wtype = getattr(getattr(pf, "projectile", None), "weapon_type", None)
    return str(wtype) if wtype else "Kinetic round"


# =============================================================================
# Since-last-checkpoint digest (ADM-06)
# =============================================================================

@dataclass
class ShipLedger:
    """Hits on one hull over an interval, by source."""
    hits: Counter = field(default_factory=Counter)      # (source_id, kind) -> count
    energy_gj: float = 0.0

    def add(self, source_id: Optional[str], kind: str, gj: float) -> None:
        self.hits[(source_id, kind)] += 1
        self.energy_gj += max(0.0, float(gj or 0.0))


@dataclass
class CheckpointDigest:
    since_t: float
    until_t: float
    kills: List[Dict[str, Any]] = field(default_factory=list)    # enemy hulls lost
    losses: List[Dict[str, Any]] = field(default_factory=list)   # ours lost
    taken: Dict[str, ShipLedger] = field(default_factory=dict)   # our hull -> hits on it
    dealt: Dict[str, ShipLedger] = field(default_factory=dict)   # enemy hull -> our hits
    torps: Dict[str, Counter] = field(default_factory=lambda: {"ours": Counter(), "theirs": Counter()})
    slugs: Dict[str, Counter] = field(default_factory=lambda: {"ours": Counter(), "theirs": Counter()})


def _event_name(event: Any) -> str:
    et = getattr(event, "event_type", None)
    return getattr(et, "name", str(et or "")).upper()


def checkpoint_digest(sim: Any, faction: str, since_t: float,
                      until_t: Optional[float] = None) -> CheckpointDigest:
    """Kills, losses, hits and ordnance outcomes in (since_t, until_t]."""
    until_t = sim.current_time if until_t is None else until_t
    digest = CheckpointDigest(since_t=since_t, until_t=until_t)

    def side(ship_id: Optional[str]) -> Optional[str]:
        f = faction_of(sim, ship_id)
        if f is None:
            return None
        return "ours" if f == faction else "theirs"

    for event in getattr(sim, "events", None) or []:
        ts = getattr(event, "timestamp", 0.0)
        if ts <= since_t or ts > until_t:
            continue
        name = _event_name(event)
        data = getattr(event, "data", None) or {}
        sid, tid = getattr(event, "ship_id", None), getattr(event, "target_id", None)
        if name in ("SHIP_DESTROYED", "SHIP_DYING"):
            entry = {"ship_id": sid, "killer_id": data.get("killer_id"), "t": ts,
                     "cause": data.get("cause") or ("crippled" if name == "SHIP_DYING" else "destroyed"),
                     "dying": name == "SHIP_DYING"}
            bucket = digest.losses if side(sid) == "ours" else digest.kills
            # A hulk that detonates later in the same interval is one loss.
            if not any(e["ship_id"] == sid for e in bucket):
                bucket.append(entry)
            elif name == "SHIP_DESTROYED":
                for e in bucket:
                    if e["ship_id"] == sid:
                        e.update(dying=False, cause=entry["cause"], t=ts)
        elif name == "TORPEDO_LAUNCHED" and side(sid):
            digest.torps[side(sid)]["launched"] += 1
        elif name == "TORPEDO_IMPACT":
            s = side(sid)
            if s:
                digest.torps[s]["impact"] += 1
            gj = data.get("total_damage_gj", 0.0)
            if side(tid) == "ours":
                digest.taken.setdefault(tid, ShipLedger()).add(sid, "torpedo", gj)
            elif side(tid) == "theirs":
                digest.dealt.setdefault(tid, ShipLedger()).add(sid, "torpedo", gj)
        elif name == "PROJECTILE_LAUNCHED" and side(sid):
            digest.slugs[side(sid)]["fired"] += 1
        elif name == "PROJECTILE_IMPACT":
            s = side(sid)
            if s:
                digest.slugs[s]["hit"] += 1
            gj = data.get("kinetic_energy_gj", 0.0)
            if side(tid) == "ours":
                digest.taken.setdefault(tid, ShipLedger()).add(sid, "slug", gj)
            elif side(tid) == "theirs":
                digest.dealt.setdefault(tid, ShipLedger()).add(sid, "slug", gj)
        elif name in ("PD_TORPEDO_DISABLED", "PD_TORPEDO_DESTROYED"):
            s = side(data.get("source_ship"))
            if s:
                digest.torps[s]["blinded" if name == "PD_TORPEDO_DISABLED" else "destroyed"] += 1
        elif name == "TORPEDO_FUEL_EXHAUSTED" and side(sid):
            digest.torps[side(sid)]["dry"] += 1
        elif name == "TORPEDO_RETARGETED" and side(sid):
            digest.torps[side(sid)]["retargeted"] += 1
        elif name == "PROJECTILE_MISS" and data.get("type") == "torpedo" and side(sid):
            digest.torps[side(sid)]["missed"] += 1
    return digest


def damage_ledger(sim: Any, faction: str) -> Dict[str, Dict[str, float]]:
    """
    Cumulative damage each ENEMY hull has done to us this battle:
    {enemy_id: {"hits": n, "gj": energy, "torpedoes_launched": k}}.
    """
    ledger: Dict[str, Dict[str, float]] = {}
    for event in getattr(sim, "events", None) or []:
        name = _event_name(event)
        if name not in ("TORPEDO_IMPACT", "PROJECTILE_IMPACT", "TORPEDO_LAUNCHED"):
            continue
        sid, tid = getattr(event, "ship_id", None), getattr(event, "target_id", None)
        if not sid or faction_of(sim, sid) == faction:
            continue
        row = ledger.setdefault(sid, {"hits": 0, "gj": 0.0, "torpedoes_launched": 0})
        data = getattr(event, "data", None) or {}
        if name == "TORPEDO_LAUNCHED":
            row["torpedoes_launched"] += 1
        elif faction_of(sim, tid) == faction:
            row["hits"] += 1
            row["gj"] += float(data.get("total_damage_gj", data.get("kinetic_energy_gj", 0.0)) or 0.0)
    return ledger


# =============================================================================
# Fleet frame
# =============================================================================

def centre(ships: Sequence[Any]) -> Tuple[Vector3D, Vector3D]:
    """(mean position, mean velocity) of a group."""
    n = max(1, len(ships))
    pos = Vector3D(sum(s.position.x for s in ships) / n, sum(s.position.y for s in ships) / n,
                   sum(s.position.z for s in ships) / n)
    vel = Vector3D(sum(s.velocity.x for s in ships) / n, sum(s.velocity.y for s in ships) / n,
                   sum(s.velocity.z for s in ships) / n)
    return pos, vel


def direction_str(v: Vector3D) -> str:
    """'(+0.99,-0.12,+0.00)' unit direction, or '(at rest)'."""
    if v.magnitude < 1e-6:
        return "(at rest)"
    u = v.normalized()
    return f"({u.x:+.2f},{u.y:+.2f},{u.z:+.2f})"


def motion_split(own_pos: Vector3D, own_vel: Vector3D, en_pos: Vector3D, en_vel: Vector3D,
                 com_vel: Vector3D) -> Tuple[float, float]:
    """
    Who is doing the closing: each side's velocity in the battle's centre-of-
    mass frame, projected on the line between them. Returns (ours toward
    them, theirs toward us) in m/s; their sum is the closing speed. Fixes the
    symmetric "Enemy fleeing" both admirals were told after a pass.
    """
    rel = en_pos - own_pos
    if rel.magnitude < 1.0:
        return 0.0, 0.0
    los = rel.normalized()
    ours = (own_vel - com_vel).dot(los)
    theirs = -(en_vel - com_vel).dot(los)
    return ours, theirs


# =============================================================================
# Engine execution notes
# =============================================================================

# Execution modes that mean "the engine is dodging something" (or, WOBBLE,
# EVADE with nothing to dodge: weaving on the current course).
EVASION_MODES = {
    "RUN": "running from a guided torpedo",
    "PRESENT": "presenting armor to an unavoidable torpedo",
    "EVADE": "jinking off a slug's line",
    "TANK": "tanking a slug on armor",
    "BRACE": "bracing",
    "WOBBLE": "EVADE with nothing to dodge: weaving on its course",
}
# Weapon states that are not a problem worth an admiral's attention.
_WEAPON_OK = {"fired", "cooldown", "reloading", "ready", "salvo_cooldown", "charging"}


def engine_notes(sim: Any, ship_ids: Iterable[str], since_t: float,
                 label_of: Any = None, limit: int = 8) -> List[str]:
    """
    What the engine actually did with our ships' orders, compactly: helm
    modes that differ from a plain burn (evasion), guns held by a gate, and
    commands rejected/cancelled/expired since ``since_t`` (ADM-07g). Built
    from the same WEAPON_STATUS / EXECUTION_STATE / COMMAND_STATUS events the
    captains' execution feedback reads.
    """
    ids = set(ship_ids)
    name_of = label_of or (lambda sid: sid)
    helm: Dict[str, Dict[str, Any]] = {}
    weapons: Dict[str, Dict[str, str]] = {}
    problems: Dict[str, Counter] = {}
    for event in getattr(sim, "events", None) or []:
        sid = getattr(event, "ship_id", None)
        if sid not in ids:
            continue
        name = _event_name(event)
        data = getattr(event, "data", None) or {}
        if name == "EXECUTION_STATE":
            helm[sid] = data
        elif name == "WEAPON_STATUS":
            weapons.setdefault(sid, {})[str(data.get("weapon_slot", "?"))] = str(data.get("reason", ""))
        elif (name == "COMMAND_STATUS" and getattr(event, "timestamp", 0.0) > since_t
              and data.get("status") in ("rejected", "cancelled", "expired")):
            command = data.get("command") or {}
            kind = (command.get("type") or command.get("maneuver_type") or "command"
                    if isinstance(command, dict) else "command")
            reason = str(data.get("reason") or data.get("status"))
            problems.setdefault(sid, Counter())[f"{kind} {data.get('status')} ({reason})"] += 1
    lines = []
    for sid in sorted(ids, key=lambda x: (len(x), x)):
        parts = []
        mode = str((helm.get(sid) or {}).get("mode") or "").upper()
        if mode in EVASION_MODES:
            parts.append(f"helm {mode} ({EVASION_MODES[mode]})")
        blocked = Counter(r for r in (weapons.get(sid) or {}).values()
                          if r and r not in _WEAPON_OK)
        if blocked:
            parts.append("guns held: " + ", ".join(
                f"{reason}" + (f" x{n}" if n > 1 else "") for reason, n in blocked.most_common()))
        for text, n in (problems.get(sid) or Counter()).items():
            parts.append(text + (f" x{n}" if n > 1 else ""))
        if parts:
            lines.append(f"{name_of(sid)}: " + "; ".join(parts))
    if len(lines) > limit:
        lines = lines[:limit] + [f"(+{len(lines) - limit} more ships)"]
    return lines


# =============================================================================
# Armament summaries
# =============================================================================

_WEAPON_SHORT = {
    "spinal_coiler_mk3": "spinal",
    "heavy_siege_coiler_mk3": "siege spinal",
    "heavy_coilgun_mk3": "heavy coil",
    "coilgun_mk3": "coil",
    "light_coilgun_mk3": "light coil",
    "torpedo_launcher": "torpedo tube",
}


def class_armament(ship_type: str, fleet_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Class armament from fleet data: {"summary": "1 spinal + 2 coil",
    "launchers": n, "magazine": rounds, "pd": n, "gun_range_km": max}.
    Every weapon type is named - torpedo launchers too (they used to be
    dropped from the enemy armament line, R2).
    """
    spec = ((fleet_data or {}).get("ships") or {}).get(ship_type) or {}
    wtypes = (fleet_data or {}).get("weapon_types") or {}
    counts: Counter = Counter()
    launchers = magazine = pd = 0
    gun_range = 0.0
    for w in spec.get("weapons", []):
        wtype = w.get("type", "")
        if wtype.startswith("pd_"):
            pd += 1
            continue
        if wtype.startswith("torpedo"):
            launchers += 1
            magazine += int(w.get("magazine", wtypes.get(wtype, {}).get("magazine", 8)))
        else:
            gun_range = max(gun_range, float(wtypes.get(wtype, {}).get("range_km", 0.0) or 0.0))
        counts[_WEAPON_SHORT.get(wtype, wtypes.get(wtype, {}).get("name", wtype))] += 1
    parts = []
    for name, n in counts.items():
        plural = "s" if n > 1 and name.endswith("tube") else ""
        parts.append(f"{n} {name}{plural}")
    return {"summary": " + ".join(parts) if parts else "unarmed", "launchers": launchers,
            "magazine": magazine, "pd": pd, "gun_range_km": gun_range}
