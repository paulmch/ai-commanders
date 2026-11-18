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
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from ..physics import G_STANDARD, Vector3D
    from ..pointdefense import PDLaser, TORPEDO_ELECTRONICS_THRESHOLD_J
    from ..torpedo import (
        ASSUMED_TARGET_MAX_LATERAL_ACCEL_MS2, MIN_CLOSING_SPEED_KPS, NEZ_DV_RESERVE_KPS,
        RESERVE_CRUISE_CLOSING_KPS, RESERVE_TERMINAL_WINDOW_S,
        NEZ_GUIDANCE_MARGIN, TERMINAL_MIN_BURN_KPS, TorpedoGuidance,
    )
except ImportError:  # pragma: no cover - script-style imports
    from physics import G_STANDARD, Vector3D
    from pointdefense import PDLaser, TORPEDO_ELECTRONICS_THRESHOLD_J
    from torpedo import (
        ASSUMED_TARGET_MAX_LATERAL_ACCEL_MS2, MIN_CLOSING_SPEED_KPS, NEZ_DV_RESERVE_KPS,
        RESERVE_CRUISE_CLOSING_KPS, RESERVE_TERMINAL_WINDOW_S,
        NEZ_GUIDANCE_MARGIN, TERMINAL_MIN_BURN_KPS, TorpedoGuidance,
    )

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

# Rounds a target needs to die and the per-round energy they were measured at:
# ONE engine-side estimate (src/torpedo_link.py) shared by the torpedo
# datalink, the captain's fire solution and this admiral economy.
try:
    from ..torpedo_link import (  # noqa: F401  (re-exported)
        KILL_ROUNDS_LIGHT, KILL_ROUNDS_HEAVY, LIGHT_HULLS, KILL_ROUNDS_REF_GJ,
        kill_rounds, kill_rounds_at,
    )
    from .. import torpedo_link as _torpedo_link
except ImportError:  # pragma: no cover - script-style imports
    from torpedo_link import (  # noqa: F401
        KILL_ROUNDS_LIGHT, KILL_ROUNDS_HEAVY, LIGHT_HULLS, KILL_ROUNDS_REF_GJ,
        kill_rounds, kill_rounds_at,
    )
    import torpedo_link as _torpedo_link


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


_LABEL_RE = re.compile(r"^(.*?)\s*\[([^\[\]]+)\]\s*$")


def split_label(text: Any) -> Tuple[str, Optional[str]]:
    """'OCS Lancer-1 [beta_2]' -> ('OCS Lancer-1', 'beta_2'); a bare name or id -> (it, None)."""
    text = " ".join(str(text or "").split())
    match = _LABEL_RE.match(text)
    if not match:
        return text, None
    return match.group(1).strip(), match.group(2).strip()


def resolve_ship_ref(text: Any, names: Dict[str, str]) -> Optional[str]:
    """
    The ship id a reference means, or None: a 'NAME [id]' label (the id
    decides), a bare id, or a full name (case-insensitive). ``names`` maps
    ship id -> ship name. Admirals copy the turn state's labels verbatim
    into tool arguments, so every consumer must accept that form too.
    """
    name, sid = split_label(text)
    if sid and sid in names:
        return sid
    key = (name or sid or "").lower()
    if not key:
        return None
    for ship_id, ship_nm in names.items():
        if key == ship_id.lower() or key == " ".join(str(ship_nm or "").split()).lower():
            return ship_id
    return None


def plain_ship_ref(text: Any, names: Dict[str, str]) -> Any:
    """
    An order field in the form the rest of the stack matches on: a
    'NAME [id]' label becomes that ship's full name (resolved through the id,
    so a mislabelled pair still lands on the right hull); anything else is
    returned unchanged. Captain prompts, the comms layer's compliance check
    and the recorder all match bare names and ids, not labels.
    """
    if not isinstance(text, str):
        return text
    name, sid = split_label(text)
    if sid is None:
        return text
    if sid in names:
        return names[sid] or sid
    return name or sid


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
    ``cross_kps`` to seeker-kill it (src/torpedo_link.turrets_to_blind: the
    continuous-dwell closed form the PD doctrine quotes).
    """
    return _torpedo_link.turrets_to_blind(cross_kps, fleet_data)


def pd_range_m(fleet_data: Optional[Dict[str, Any]]) -> float:
    spec = ((fleet_data or {}).get("weapon_types") or {}).get("pd_laser") or {}
    return float(spec.get("range_km", 250.0)) * 1000.0


def pd_cover(sim: Any, target: Any, radius_m: float, threat_from: Any = None) -> int:
    """Operational PD turrets of the target's side within ``radius_m`` of it.
    With ``threat_from`` (where the rounds come from) only turrets that BEAR
    on that point are counted (weapon arcs, src/weapon_arcs.py)."""
    from .arcs_status import pd_turrets_bearing
    total = 0
    for ship in getattr(sim, "ships", {}).values():
        if ship.faction != target.faction or not is_live(ship):
            continue
        if (ship.position - target.position).magnitude <= radius_m:
            total += (pd_turrets_bearing(sim, ship, threat_from)[0]
                      if threat_from is not None else pd_turrets(ship)[0])
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
    target_delay_s: float = 0.0,
    profile: str = "sprint",
) -> Dict[str, Any]:
    """
    Fly a 1-D pursuit along the line of sight, worst case for the torpedo:
    the target RUNS (full acceleration straight away) for the whole flight.

    The torpedo first spends whatever nulls the crossing (lateral) speed, then
    flies the engine's own throttle law (TorpedoGuidance._collision_course_guidance)
    step by step:

    - below the closure floor (MIN_CLOSING_SPEED_KPS): full thrust;
    - at or above it: the terminal burn, which only ever spends the
      no-escape-zone SURPLUS - delta-v beyond what a worst-case 3 g jink could
      still demand over the remaining time to go
      (NEZ_GUIDANCE_MARGIN * a_t * t_go + NEZ_DV_RESERVE_KPS, nez_status) -
      and only once that surplus clears TERMINAL_MIN_BURN_KPS.

    The surplus grows as t_go falls, so a long flight dumps nearly all its
    delta-v into closure, while a short-range launch arrives still climbing
    toward the floor with most of its delta-v unspent: thrust the round never
    had time to fire is never counted. Thrust is the specs' thrust over the
    falling mass (rocket equation), the same drive the engine integrates.

    Returns hit (bool), tof_s, impact_mps, cross_mps (mean closure while
    inside the PD envelope - what PD dwell depends on), short_m.
    ``target_delay_s``: the target starts running only then (it turns first).
    ``profile`` 'reserve': the reserve launch profile's cruise floor, and the
    terminal burn only inside its window (src/torpedo.py).
    """
    window_s = None
    if profile == "reserve":
        floor_mps = RESERVE_CRUISE_CLOSING_KPS * 1000.0
        window_s = RESERVE_TERMINAL_WINDOW_S
    ve = max(1.0, float(specs.exhaust_velocity_kps) * 1000.0)
    m = float(mass_kg if mass_kg is not None else specs.mass_kg)
    dv = (float(dv_left_mps) if dv_left_mps is not None
          else ve * math.log(max(1.0, m / max(1.0, specs.dry_mass_kg))))
    dv -= max(0.0, lateral_mps)
    if dv < 0:
        return {"hit": False, "short_m": range_m, "reason": "crossing too fast"}
    a_t = ASSUMED_TARGET_MAX_LATERAL_ACCEL_MS2
    reserve = NEZ_DV_RESERVE_KPS * 1000.0
    min_burn = TERMINAL_MIN_BURN_KPS * 1000.0
    c, d, t = closing_mps, range_m, 0.0
    dry_t: Optional[float] = None   # when the round's delta-v ran out (None: never)
    inside_t = 0.0
    inside_start_d = min(d, pd_envelope_m)
    dt = SOLUTION_STEP_S
    while t < horizon_s:
        burn = 0.0
        a = specs.thrust_n / max(1.0, m)
        if dv > 0:
            if c < floor_mps:
                burn = min(a * dt, dv)
            elif d > 0 and (window_s is None or d / c <= window_s):
                surplus = dv - (NEZ_GUIDANCE_MARGIN * a_t * (d / c) + reserve)
                if surplus > min_burn and a >= NEZ_GUIDANCE_MARGIN * a_t:
                    burn = min(a * dt, surplus)
        if burn > 0:
            c += burn
            dv -= burn
            m *= math.exp(-burn / ve)
            if dv <= 1e-6 and dry_t is None:
                dry_t = t
        if t >= target_delay_s:
            c -= target_accel_ms2 * dt
        d -= c * dt
        t += dt
        if d <= pd_envelope_m:
            inside_t += dt
        if d <= 0:
            cross = inside_start_d / inside_t if inside_t > 0 else c
            return {"hit": True, "tof_s": t, "impact_mps": c, "cross_mps": cross,
                    "dry_before_impact": dry_t is not None}
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
    profile: str = "sprint",
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
                        mass_kg=mass_kg, dv_left_mps=dv_left_mps, pd_envelope_m=envelope,
                        profile=profile)
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
        # turrets that bear on a round coming from the launch point (weapon arcs)
        sol.pd_cover = pd_cover(sim, target, envelope, threat_from=launch_pos)
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
            # The round keeps burning (closure floor, terminal burn): its ETA
            # is the guidance law's flight time against a target that holds
            # course (solve_pursuit), not range / closing now - which read
            # "ETA 84 s" for a salvo that landed in 40.
            try:
                fly = solve_pursuit(dist, closing, 0.0, 0.0, torp.specs,
                                    mass_kg=torp.current_mass_kg,
                                    dv_left_mps=info.dv_left_kps * 1000.0)
                if not fly.get("hit"):
                    fly = None
                else:
                    eta = fly["tof_s"]
            except Exception:  # noqa: BLE001 - advisory only
                fly = None
            info.range_km, info.closing_kps, info.eta_s = dist / 1000.0, closing / 1000.0, eta
            # Projected impact: current relative speed plus the closure the
            # remaining burn can still add (the engine's threat estimate).
            if eta is not None:
                gain = min(info.dv_left_kps * 1000.0,
                           torp.specs.acceleration_at_mass(torp.current_mass_kg) * eta)
            else:
                gain = info.dv_left_kps * 1000.0
            speed = rel_vel.magnitude + max(0.0, gain)
            if fly is not None:
                speed = fly["impact_mps"]   # the same forecast as the ETA
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
    """
    Kills, losses, hits and ordnance outcomes in [since_t, until_t).

    Half-open at the END, not the start: a sim step logs its events at the
    step's start time, and the decisions at checkpoint T run before the step
    that begins at T. So an event stamped T (every launch the captains
    ordered at T, the first salvo after it) happened AFTER the checkpoint-T
    decision and belongs to the window that starts at T; nothing stamped
    until_t has happened yet when the admiral decides at until_t.
    """
    until_t = sim.current_time if until_t is None else until_t
    digest = CheckpointDigest(since_t=since_t, until_t=until_t)

    def side(ship_id: Optional[str]) -> Optional[str]:
        f = faction_of(sim, ship_id)
        if f is None:
            return None
        return "ours" if f == faction else "theirs"

    for event in getattr(sim, "events", None) or []:
        ts = getattr(event, "timestamp", 0.0)
        if ts < since_t or ts >= until_t:
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


def motion_split(own_pos: Vector3D, own_vel: Vector3D, en_pos: Vector3D,
                 en_vel: Vector3D) -> Tuple[float, float]:
    """
    Who is doing the closing: each side's velocity in the battle frame (the
    simulation's world frame - fleets start the battle at rest in it unless a
    config gives them a velocity), projected on the line between them.
    Returns (ours toward them, theirs toward us) in m/s; their sum is the
    closing speed. Fixes the symmetric "Enemy fleeing" both admirals were
    told after a pass.

    Not the two fleets' centre-of-mass frame: there the split depends only on
    the ship counts, never on who moved (one ship each always splits 50/50,
    whether we charged a sitting enemy or he charged us).
    """
    rel = en_pos - own_pos
    if rel.magnitude < 1.0:
        return 0.0, 0.0
    los = rel.normalized()
    return own_vel.dot(los), -en_vel.dot(los)


def burn_toward_g(own_pos: Vector3D, en_pos: Vector3D, accels: Sequence[Vector3D]) -> float:
    """
    An enemy group's mean current thrust acceleration along the line toward
    us, in g (negative = burning away). Velocity says who HAS closed; the
    burn says who is closing NOW - a fleet braking after a charge still shows
    its old speed toward us while its torch points at us.
    """
    rel = own_pos - en_pos
    if rel.magnitude < 1.0 or not accels:
        return 0.0
    los = rel.normalized()
    return sum(a.dot(los) for a in accels) / len(accels) / G_STANDARD


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
    commands rejected/cancelled/expired at or after ``since_t`` (ADM-07g; a
    rejection of an order issued at the previous decision is stamped with
    that decision's time, see checkpoint_digest). Built
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
        elif (name == "COMMAND_STATUS" and getattr(event, "timestamp", 0.0) >= since_t
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
        if (helm.get(sid) or {}).get("swing") and mode in EVASION_MODES:
            # The evade overlay keeping a burning order's burn and swinging
            # it (simulation module docstring), not taking the drive.
            parts.append(f"helm {mode} (swinging its ordered burn within 20 deg of the line)")
        elif mode in EVASION_MODES:
            parts.append(f"helm {mode} ({EVASION_MODES[mode]})")
        blocked = Counter(r for r in (weapons.get(sid) or {}).values()
                          if r and r not in _WEAPON_OK)
        if blocked:
            parts.append("guns held: " + ", ".join(
                f"{reason}" + (f" x{n}" if n > 1 else "") for reason, n in blocked.most_common()))
        for text, n in (problems.get(sid) or Counter()).items():
            parts.append(text + (f" x{n}" if n > 1 else ""))
        if parts:
            lines.append((sid, parts))
    # Ships whose only note is the same gun hold share one line ("guns held:
    # probability_gate" repeated for 8 ships was 8 lines - audit A7).
    shared: Dict[str, List[str]] = {}
    for sid, parts in lines:
        if len(parts) == 1 and parts[0].startswith("guns held: "):
            shared.setdefault(parts[0], []).append(sid)
    out = []
    done = set()
    for sid, parts in lines:
        key = parts[0] if len(parts) == 1 else None
        if key in shared and len(shared[key]) > 1:
            if key not in done:
                done.add(key)
                out.append(f"{key} ({len(shared[key])} ships): "
                           + ", ".join(name_of(s) for s in shared[key]))
            continue
        out.append(f"{name_of(sid)}: " + "; ".join(parts))
    lines = out
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


# =============================================================================
# Facing: what face a ship shows a point, and what a heading does to it
# =============================================================================

def _hit_cones() -> Tuple[float, float]:
    """The engine's hit-location cones (src/geometry.py): a round arriving
    within 30 deg of the nose hits the nose, beyond 150 deg the tail.
    Imported lazily: geometry.py only imports as part of the src package."""
    try:
        from ..geometry import NOSE_HIT_ANGLE_THRESHOLD, TAIL_HIT_ANGLE_THRESHOLD
    except ImportError:  # pragma: no cover - script-style imports
        from geometry import NOSE_HIT_ANGLE_THRESHOLD, TAIL_HIT_ANGLE_THRESHOLD
    return NOSE_HIT_ANGLE_THRESHOLD, TAIL_HIT_ANGLE_THRESHOLD

# Below this applied throttle the torch is PADLOCK's token burn (5% while
# turning), not a burn - the admiral's plot used to call that "burning".
BURNING_MIN_THROTTLE = 0.1
# Spin above this counts as turning (a thrust-vectored turn runs 4-15 deg/s).
TURNING_MIN_DEG_S = 0.5
_FACE_SECTION = {"NOSE": "nose", "SIDE": "lateral", "TAIL": "tail"}


def face_of(angle_deg: float) -> str:
    """NOSE / SIDE / TAIL for an angle off the nose (the engine's hit cones)."""
    nose, tail = _hit_cones()
    if angle_deg < nose:
        return "NOSE"
    if angle_deg > tail:
        return "TAIL"
    return "SIDE"


def is_burning(ship: Any) -> bool:
    """The drive is producing thrust (not PADLOCK's token burn)."""
    return float(getattr(ship, "applied_throttle", 0.0) or 0.0) >= BURNING_MIN_THROTTLE \
        and ship_accel_g(ship) > 0.0


def ship_accel_g(ship: Any) -> float:
    ks = getattr(ship, "kinematic_state", None)
    if ks is None or not getattr(ks, "mass_kg", 0):
        return 0.0
    return float(getattr(ks, "thrust_n", 0.0)) / ks.mass_kg / G_STANDARD


def observed_facing(ship: Any, toward: Vector3D) -> Optional[Dict[str, Any]]:
    """
    The face ``ship`` shows a point (an observer): NOSE / SIDE / TAIL by the
    engine's hit cones, that facing's live armour, and whether it is burning
    or turning - all observable from outside (attitude, torch, spin), never
    its commanded maneuver. None without attitude data.
    """
    fwd = getattr(ship, "forward", None)
    if fwd is None or fwd.magnitude < 1e-9:
        return None
    los = toward - ship.position
    if los.magnitude < 1.0:
        return None
    angle = math.degrees(fwd.angle_to(los))
    face = face_of(angle)
    armor_cm = None
    armor = getattr(ship, "armor", None)
    if armor is not None:
        try:
            from ..combat import HitLocation
        except ImportError:  # pragma: no cover
            from combat import HitLocation
        sec = armor.get_section(HitLocation(_FACE_SECTION[face]))
        if sec is not None:
            armor_cm = float(sec.thickness_cm)
    spin = math.degrees(float(getattr(ship, "angular_velocity_rad_s", 0.0) or 0.0))
    return {"face": face, "angle_deg": angle, "armor_cm": armor_cm,
            "burning": is_burning(ship), "turning": spin >= TURNING_MIN_DEG_S}


def facing_text(f: Optional[Dict[str, Any]]) -> str:
    """'SIDE 30cm, burning' - the ~5-token form for threat rows."""
    if not f:
        return ""
    text = f["face"] + (f" {f['armor_cm']:.0f}cm" if f.get("armor_cm") is not None else "")
    motion = [w for w, on in (("burning", f.get("burning")), ("turning", f.get("turning"))) if on]
    return text + ", " + ("+".join(motion) if motion else "coasting")


def _median(values: List[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2] if values else 0.0


def him_after_text(sim: Any, info: Any, own: Sequence[Any], enemies: Sequence[Any],
                   opt: Any) -> str:
    """
    'him: NOSE 11->SIDE 84 deg (re-points in ~20 s)': the face the enemy group
    (median member) shows your centre now, and once this option has flown
    (its turn + burn; the coast option over at most 60 s) if he holds his
    attitude and course, with the time his ship needs to re-point at you.
    The option rows priced only your own exposure (audit A3/A4).
    """
    own_pos, own_vel = centre(own)
    accel = float(getattr(info, "accel_g", 0.0) or 0.0) * G_STANDARD
    h = opt.heading
    thr = float(opt.throttle or 0.0)
    burn = float(opt.burn_s or 0.0) if thr > 0 else 0.0
    t_end = float(opt.turn_s or 0.0) + burn
    if opt.key == "face":
        t_end = min(t_end, 60.0)
    moved = own_pos + own_vel * t_end
    if burn > 0:
        moved = moved + h * (0.5 * accel * thr * burn * burn)
    angles_now = [math.degrees(e.forward.angle_to(own_pos - e.position)) for e in enemies]
    now = _median(angles_now)
    after = []
    for e in enemies:
        e_end = e.position + e.velocity * t_end
        after.append((math.degrees(e.forward.angle_to(moved - e_end)), e, moved - e_end))
    after.sort(key=lambda x: x[0])
    angle_after, rep, los_after = after[len(after) // 2]
    text = f"him: {face_of(now)} {now:.0f}->{face_of(angle_after)} {angle_after:.0f} deg"
    faces = [face_of(a) for a in angles_now]
    if len(set(faces)) > 1:
        # A mixed group (say 3 nose-on frigates and 5 corvettes tail-on): the
        # median member alone reads as if the whole group showed that face.
        split = ", ".join(f"{faces.count(f)} {f}" for f in ("NOSE", "SIDE", "TAIL") if f in faces)
        text = f"him (median of {len(faces)}: {split} now): " + text[len("him: "):]
    if face_of(angle_after) != "NOSE":
        try:
            turn = sim._turn_time_to(rep, los_after, True)
            if math.isfinite(turn):
                text += f" (re-points in ~{turn:.0f} s)"
        except Exception:  # noqa: BLE001 - advisory only
            pass
    return text


def heading_option_rows(sim: Any, info: Any, own: Sequence[Any], enemies: Sequence[Any],
                        verb: str = "heading") -> List[str]:
    """Range-control option rows (range_control's own rendering: range effect,
    the face you show and what bears) with the face HE shows you appended."""
    from .range_control import format_heading_options
    text = format_heading_options([info], max_detail=99, verb=verb)
    return [line for line in annotate_option_rows(text, sim, [info], own).splitlines()
            if line.startswith("  ")]


def annotate_option_rows(text: str, sim: Any, infos: Sequence[Any], own: Sequence[Any]) -> str:
    """Append ' | him: ...' (him_after_text) to every option row of a
    range-control text block, matching rows to options by their heading
    vector within the current group (merged rows such as
    'cancel closure = open range' keep range_control's wording)."""
    if not text or not own:
        return text
    labels = {info.label: info for info in infos}
    current = infos[0] if len(infos) == 1 else None
    enemies: List[Any] = []
    out: List[str] = []

    def members(info: Any) -> List[Any]:
        return [x for x in (sim.get_ship(sid) for sid in info.ship_ids) if x is not None and is_live(x)]
    if current is not None:
        enemies = members(current)
    for line in text.splitlines():
        head = line.split(" ", 1)[0].rstrip(":") if not line.startswith(" ") else None
        if head is not None:
            key = next((k for k in labels if line.startswith(k + " ") or line.startswith(k + ":")), None)
            if key is not None:
                current = labels[key]
                enemies = members(current)
        if current is not None and enemies and line.startswith("  ") and "heading(" in line:
            opt = next((o for o in current.options if f"heading{o.heading_str()}" in line), None)
            if opt is not None:
                try:
                    line += " | " + him_after_text(sim, current, own, enemies, opt)
                except Exception:  # noqa: BLE001 - advisory only
                    pass
        out.append(line)
    return "\n".join(out)


def admiral_headings_block(sim: Any, infos: Sequence[Any], own: Sequence[Any]) -> str:
    """The HEADINGS body for the admiral's directive (fleet centre):
    range_control.format_heading_options with the face he shows you added."""
    from .range_control import format_heading_options
    return annotate_option_rows(format_heading_options(infos), sim, infos, own)


def replace_option_rows(text: str, sim: Any, infos: Sequence[Any], own: Sequence[Any]) -> str:
    """The per-ship order's range-control block with the face he shows you added."""
    return annotate_option_rows(text, sim, infos, own)


# =============================================================================
# The clock and the time-limit verdict
# =============================================================================

def battle_clock(time_limit_s: Optional[float], max_checkpoints: Optional[int],
                 interval_s: float, unlimited: bool = False,
                 last_orders_fly: bool = False) -> Optional[Dict[str, Any]]:
    """
    When the battle ends and how many checkpoints it has, as the runner flies
    it: checkpoints fall every ``interval_s`` from T+interval while the time
    limit has not been reached, at most ``max_checkpoints``; the runner stops
    after the last checkpoint's decisions unless ``last_orders_fly`` (the live
    draft runner gives them one more window). None in unlimited mode.
    """
    if unlimited or not interval_s or interval_s <= 0:
        return None
    limit = float(time_limit_s) if time_limit_s else float("inf")
    by_time = max(0, math.ceil(limit / interval_s) - 1) if math.isfinite(limit) else None
    candidates = [c for c in (max_checkpoints, by_time) if c is not None]
    if not candidates:
        return None
    checkpoints = int(min(candidates))
    if by_time is not None and (max_checkpoints is None or by_time < max_checkpoints):
        # The time limit binds: the last checkpoint's orders fly until it.
        end = limit
    else:
        end = checkpoints * interval_s + (interval_s if last_orders_fly else 0.0)
    return {"checkpoints": checkpoints, "end_s": min(limit, end), "interval_s": float(interval_s)}


def clock_text(now_s: float, checkpoint: int, clock: Optional[Dict[str, Any]]) -> str:
    """', checkpoint 7 of 20, battle ends at T+630s' ('' without a clock)."""
    if not clock or not checkpoint:
        return ""
    return (f", checkpoint {checkpoint} of {clock['checkpoints']}, battle ends at "
            f"T+{clock['end_s']:.0f}s")


def time_limit_scores(alpha_ships: Iterable[Any], beta_ships: Iterable[Any]) -> Tuple[float, float]:
    """
    The fleet time-limit verdict's two scores (pure): each surviving hull
    scores its draft value x hull fraction x the disarmed/adrift factor, plus
    0.1 per GJ it dealt, plus 10 x its hit rate; each lost hull gives the
    other side 1.5 x its value, a surrendered one 0.75 x. The additions run
    in exactly the order battle_runner._evaluate_fleet_tactical_advantage
    always used, so the floats - and the recorded reason string - are
    unchanged.
    """
    try:
        from .battle_runner import ship_class_value, surviving_ship_score_factor
    except ImportError:  # pragma: no cover - script-style imports
        from llm.battle_runner import ship_class_value, surviving_ship_score_factor
    alpha_score = 0.0
    beta_score = 0.0
    for ship in alpha_ships:
        base_value = ship_class_value(getattr(ship, 'ship_type', 'destroyer'))
        if ship.is_destroyed or getattr(ship, 'is_dying', False):
            beta_score += base_value * 1.5
        elif getattr(ship, 'is_surrendered', False):
            beta_score += base_value * 0.75
        else:
            hull_pct = ship.hull_integrity / 100.0
            alpha_score += base_value * hull_pct * surviving_ship_score_factor(ship)
            alpha_score += ship.damage_dealt_gj * 0.1
            if ship.shots_fired > 0:
                accuracy = ship.hits_scored / ship.shots_fired
                alpha_score += accuracy * 10
    for ship in beta_ships:
        base_value = ship_class_value(getattr(ship, 'ship_type', 'destroyer'))
        if ship.is_destroyed or getattr(ship, 'is_dying', False):
            alpha_score += base_value * 1.5
        elif getattr(ship, 'is_surrendered', False):
            alpha_score += base_value * 0.75
        else:
            hull_pct = ship.hull_integrity / 100.0
            beta_score += base_value * hull_pct * surviving_ship_score_factor(ship)
            beta_score += ship.damage_dealt_gj * 0.1
            if ship.shots_fired > 0:
                accuracy = ship.hits_scored / ship.shots_fired
                beta_score += accuracy * 10
    return alpha_score, beta_score


def score_line(sim: Any, faction: str) -> str:
    """'SCORE IF THE BATTLE ENDED NOW (time-limit verdict): you 212 / them 148'
    - shown only when AdmiralConfig.show_score is on (off by default)."""
    ships = list((getattr(sim, "ships", None) or {}).values())
    alpha = [s for s in ships if getattr(s, "faction", None) == "alpha"]
    beta = [s for s in ships if getattr(s, "faction", None) == "beta"]
    a, b = time_limit_scores(alpha, beta)
    you, them = (a, b) if faction == "alpha" else (b, a)
    # The verdict the runner draws from these scores (battle_runner
    # _evaluate_fleet_tactical_advantage): closer than max(5, 5% of the
    # higher) is a draw - without it "you 151 / them 145" reads as a win.
    margin = max(TIME_LIMIT_DRAW_MIN_POINTS, max(you, them) * TIME_LIMIT_DRAW_FRACTION)
    verdict = ("a DRAW (closer than the margin, " f"{margin:.0f})" if abs(you - them) < margin
               else "you win" if you > them else "he wins")
    return (f"SCORE IF THE BATTLE ENDED NOW (time-limit verdict): you {you:.0f} / them {them:.0f}"
            f" = {verdict}")


# The time-limit verdict's draw margin: max(5 points, 5% of the higher score).
TIME_LIMIT_DRAW_MIN_POINTS = 5.0
TIME_LIMIT_DRAW_FRACTION = 0.05


# =============================================================================
# What a ship actually flew over a window (non-compliance escalation)
# =============================================================================

def helm_summary(sim: Any, ship_ids: Iterable[str], since_t: float,
                 until_t: float) -> Dict[str, Dict[str, Any]]:
    """
    Per ship, from the engine's EXECUTION_STATE transitions over
    [since_t, until_t]: the share of the window the drive produced thrust
    (applied throttle >= BURNING_MIN_THROTTLE), the share in which the
    maneuver REQUESTED thrust but none was produced (held: evasion, a dead
    drive, a turn, the interlock), the share spent in each helm mode (and,
    under "swing", the share of those in which the evade overlay swung a
    burning order instead of holding it - the drive stayed on the order),
    and the last engine fraction (0 = drive dead).
    """
    ids = set(ship_ids)
    span = max(1e-6, until_t - since_t)
    state: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    acc: Dict[str, Dict[str, Any]] = {sid: {"burn": 0.0, "held": 0.0, "modes": Counter(),
                                            "swing": Counter(), "engine": None} for sid in ids}

    def close(sid: str, t_end: float) -> None:
        start, data = state[sid]
        dt = max(0.0, min(t_end, until_t) - max(start, since_t))
        if dt <= 0:
            return
        applied = float(data.get("applied_throttle") or 0.0)
        if applied >= BURNING_MIN_THROTTLE:
            acc[sid]["burn"] += dt
        elif float(data.get("requested_throttle") or 0.0) >= BURNING_MIN_THROTTLE:
            acc[sid]["held"] += dt      # the order asked for thrust, none came out
        acc[sid]["modes"][str(data.get("mode") or "?").upper()] += dt
        if data.get("swing") and not data.get("off_line"):
            acc[sid]["swing"][str(data.get("mode") or "?").upper()] += dt

    for event in getattr(sim, "events", None) or []:
        sid = getattr(event, "ship_id", None)
        if sid not in ids or _event_name(event) != "EXECUTION_STATE":
            continue
        t = float(getattr(event, "timestamp", 0.0) or 0.0)
        if t > until_t:
            break
        if sid in state:
            close(sid, t)
        data = getattr(event, "data", None) or {}
        state[sid] = (t, data)
        acc[sid]["engine"] = data.get("engine_fraction")
    out = {}
    for sid in ids:
        if sid in state:
            close(sid, until_t)
        a = acc[sid]
        out[sid] = {"burn_frac": a["burn"] / span, "held_frac": a["held"] / span,
                    "modes": {m: v / span for m, v in a["modes"].items()},
                    "swing": {m: v / span for m, v in a["swing"].items()},
                    "engine_fraction": a["engine"]}
    return out



# =============================================================================
# Torpedo defence: per targeted ship, what arrives and what running changes
# =============================================================================

TORPEDO_DEFENCE_MAX_SHIPS = 4


TORPEDO_SHADOW_HORIZON_S = 300.0


def fly_inbound(sim: Any, target: Any, flights: Sequence[Any], fleet_data: Optional[Dict[str, Any]],
                heading: Optional[Vector3D] = None,
                horizon_s: float = TORPEDO_SHADOW_HORIZON_S) -> Optional[Dict[str, Any]]:
    """
    The engine's own forecast for rounds already in flight: copies of
    ``flights`` fly at a copy of ``target`` on a private CombatSimulation
    (no PD - turret cover is reported separately - no plume wash, no arcs,
    armour made unbreakable so every round's arrival is seen). The target
    holds course (MAINTAIN) or, with ``heading``, turns and burns along it at
    full throttle. Returns hits, first impact (s from now), median impact
    speed / energy, rounds that never hit (dry or passed). Nothing touches
    the battle or its RNG. None when the shadow cannot be built.
    """
    import copy
    try:
        from ..simulation import (CombatSimulation, Maneuver, ManeuverType,
                                  SimulationEventType, create_ship_from_fleet_data)
    except ImportError:  # pragma: no cover
        from simulation import (CombatSimulation, Maneuver, ManeuverType,
                                SimulationEventType, create_ship_from_fleet_data)
    if not fleet_data or not getattr(target, "ship_type", None) or not flights:
        return None
    try:
        shadow = CombatSimulation(time_step=1.0, decision_interval=1e9, seed=0)
        shadow.echo = False
        shadow.plume_wash_enabled = False
        shadow.weapon_arcs = False
        tg = create_ship_from_fleet_data(
            "target", target.ship_type, "alpha", fleet_data,
            position=Vector3D(target.position.x, target.position.y, target.position.z),
            velocity=Vector3D(target.velocity.x, target.velocity.y, target.velocity.z),
            forward=Vector3D(target.forward.x, target.forward.y, target.forward.z))
        far = target.position + Vector3D(1e9, 0.0, 0.0)
        src = create_ship_from_fleet_data("source", "corvette", "beta", fleet_data, position=far,
                                          forward=Vector3D(-1.0, 0.0, 0.0))
    except Exception:  # noqa: BLE001 - advisory only
        return None
    tg.point_defense = []
    src.point_defense = []
    armor = getattr(tg, "armor", None)
    for sec in (getattr(armor, "sections", None) or {}).values():
        sec.thickness_cm = 1e7
    tg.current_maneuver = (Maneuver(ManeuverType.MAINTAIN, 0.0) if heading is None else
                           Maneuver(ManeuverType.HEADING, 0.0, throttle=1.0,
                                    heading_direction={"x": heading.x, "y": heading.y, "z": heading.z}))
    shadow.add_ship(tg)
    shadow.add_ship(src)
    for tf in flights:
        twin = copy.deepcopy(tf)
        twin.torpedo.target_id = "target"
        twin.source_ship_id = "source"
        shadow.torpedoes.append(twin)
    impacts = []
    while shadow.current_time < horizon_s and shadow.torpedoes and not tg.is_destroyed:
        for ev in shadow.step() or []:
            if ev.event_type == SimulationEventType.TORPEDO_IMPACT:
                impacts.append((shadow.current_time, float(ev.data.get("impact_speed_kps") or 0.0),
                                float(ev.data.get("total_damage_gj") or 0.0)))
    out: Dict[str, Any] = {"hits": len(impacts), "dry": len(flights) - len(impacts)}
    if impacts:
        speeds = sorted(i[1] for i in impacts)
        energies = sorted(i[2] for i in impacts)
        out.update(first_s=min(i[0] for i in impacts), impact_kps=speeds[len(speeds) // 2],
                   energy_gj=energies[len(energies) // 2])
    return out


def torpedo_defence(sim: Any, faction: str, fleet_data: Optional[Dict[str, Any]],
                    rounds: Sequence[RoundInfo]) -> Dict[str, Dict[str, Any]]:
    """
    Per own ship with live enemy torpedoes homing on it (empty when none):
    rounds inbound, first impact, PD turrets that BEAR on the rounds
    (pd_cover with weapon arcs) and how many of them those turrets can blind
    at the rounds' crossing speed (turrets_to_blind), the leakers, and the
    engine's own forecast (fly_inbound: copies of the rounds flown at a copy
    of the ship on a private CombatSimulation) for holding course versus
    turning and burning straight away: impact speed / energy, first impact,
    rounds that never hit. The 1-D solve_pursuit (the guidance throttle law
    along the line of sight) is the fallback when no shadow can be built;
    it gives the PD crossing speed either way.
    """
    out: Dict[str, Dict[str, Any]] = {}
    guided: Dict[str, List[RoundInfo]] = {}
    for r in rounds:
        if r.kind == "torpedo" and r.side == "theirs" and r.state == "guided" and r.threat_id:
            guided.setdefault(r.threat_id, []).append(r)
    if not guided:
        return out
    flights = {tf.torpedo_id: tf for tf in getattr(sim, "torpedoes", None) or []}
    envelope = pd_range_m(fleet_data)
    order = sorted(guided, key=lambda sid: min(r.eta_s or 1e9 for r in guided[sid]))
    for sid in order[:TORPEDO_DEFENCE_MAX_SHIPS]:
        ship = sim.get_ship(sid)
        if ship is None or not is_live(ship) or ship.faction != faction:
            continue
        torps = [flights[r.round_id] for r in guided[sid] if r.round_id in flights]
        if not torps:
            continue
        n = len(torps)
        cx = sum(t.torpedo.position.x for t in torps) / n
        cy = sum(t.torpedo.position.y for t in torps) / n
        cz = sum(t.torpedo.position.z for t in torps) / n
        source = Vector3D(cx, cy, cz)
        away = ship.position - source
        heading = away.normalized() if away.magnitude > 1.0 else ship.forward
        try:
            turn_s = float(sim._turn_time_to(ship, heading, True))
        except Exception:  # noqa: BLE001 - advisory only
            turn_s = 0.0
        if not math.isfinite(turn_s):
            turn_s = 0.0
        accel = max_accel_ms2(ship)
        hold, run = [], []
        for tf in torps:
            torp = tf.torpedo
            rel_pos = ship.position - torp.position
            rel_vel = ship.velocity - torp.velocity
            rng = rel_pos.magnitude
            closing = _closing(rel_pos, rel_vel) if rng > 1.0 else 0.0
            dv = float(getattr(torp, "remaining_delta_v_kps", 0.0) or 0.0) * 1000.0
            kw = dict(mass_kg=torp.current_mass_kg, dv_left_mps=dv, pd_envelope_m=envelope)
            hold.append(solve_pursuit(rng, closing, 0.0, 0.0, torp.specs, **kw))
            run.append(solve_pursuit(rng, closing, 0.0, accel, torp.specs,
                                     target_delay_s=turn_s, **kw))
        pen = float(getattr(torps[0].torpedo.specs, "penetrator_mass_kg", 250.0))

        def summary(results: List[Dict[str, Any]], maneuvering: bool) -> Dict[str, Any]:
            # A round that runs dry on the way is a straight-line wreck: it
            # still strikes a hull holding course (the PD doctrine's 17/18),
            # but misses one that burns away - the engine flew it so.
            hits = [x for x in results if x.get("hit")
                    and not (maneuvering and x.get("dry_before_impact"))]
            if not hits:
                return {"hits": 0, "dry": len(results)}
            speeds = sorted(x["impact_mps"] for x in hits)
            mid = speeds[len(speeds) // 2]
            cross = sorted(x["cross_mps"] for x in hits)[len(hits) // 2]
            return {"hits": len(hits), "dry": len(results) - len(hits),
                    "first_s": min(x["tof_s"] for x in hits), "impact_kps": mid / 1000.0,
                    "energy_gj": 0.5 * pen * mid ** 2 / 1e9, "cross_kps": cross / 1000.0}

        h, rn = summary(hold, False), summary(run, True)
        # The engine's own forecast replaces the 1-D one where it can be flown.
        shadow_hold = fly_inbound(sim, ship, torps, fleet_data)
        drive = getattr(ship, "engine_effectiveness", 1.0)
        dead = isinstance(drive, (int, float)) and drive <= 0.0
        shadow_run = None if dead else fly_inbound(sim, ship, torps, fleet_data, heading=heading)
        if shadow_hold is not None:
            h = dict(shadow_hold, cross_kps=h.get("cross_kps"))
        if shadow_run is not None:
            rn = shadow_run
        if dead:
            rn = {"hits": 0, "dry": 0, "dead_drive": True}
        cover = pd_cover(sim, ship, envelope, threat_from=source)
        per_round = turrets_to_blind(fleet_data, h.get("cross_kps") or MIN_CLOSING_SPEED_KPS)
        blind = min(n, cover // max(1, per_round))
        etas = [r.eta_s for r in guided[sid] if r.eta_s is not None]
        tail = None
        armor = getattr(ship, "armor", None)
        if armor is not None:
            try:
                from ..combat import HitLocation
            except ImportError:  # pragma: no cover
                from combat import HitLocation
            sec = armor.get_section(HitLocation.TAIL)
            tail = float(sec.thickness_cm) if sec is not None else None
        out[sid] = {"rounds": n, "first_eta_s": min(etas) if etas else None,
                    "turrets_bearing": cover, "turrets_per_round": per_round,
                    "can_blind": blind, "leakers": n - blind, "hold": h, "run": rn,
                    "run_heading": heading, "turn_s": turn_s, "tail_cm": tail}
    return out


def torpedo_defence_line(label: str, d: Dict[str, Any]) -> str:
    """One line per targeted ship (the admiral's ORDNANCE block / order call)."""
    first = (d.get("hold") or {}).get("first_s", d.get("first_eta_s"))
    text = (f"{label}: {d['rounds']} round{'s' if d['rounds'] != 1 else ''}, first impact "
            + (f"~{first:.0f} s" if first is not None else "not closing")
            + f"; {d['turrets_bearing']} PD turrets bear (~{d['turrets_per_round']} per round) -> "
            f"blind ~{d['can_blind']}, ~{d['leakers']} arrive live")
    h, rn = d["hold"], d["run"]
    if h.get("hits"):
        text += f". Holding: impact ~{h['impact_kps']:.1f} km/s (~{h['energy_gj']:.0f} GJ)"
    hd = d["run_heading"]
    run = (f". Burning away heading({hd.x:+.2f},{hd.y:+.2f},{hd.z:+.2f}) @100% (turn ~{d['turn_s']:.0f} s, "
           f"then TAIL" + (f" {d['tail_cm']:.0f} cm" if d.get("tail_cm") is not None else "")
           + " to the rounds): ")
    if rn.get("dead_drive"):
        return text + ". Burning away: impossible, drive dead (0 g)"
    if rn.get("hits"):
        bought = (rn["first_s"] - h["first_s"]) if h.get("hits") else None
        run += (f"impact ~{rn['impact_kps']:.1f} km/s (~{rn['energy_gj']:.0f} GJ), first impact "
                f"~{rn['first_s']:.0f} s" + (f" ({bought:+.0f} s)" if bought is not None else ""))
        if rn.get("dry"):
            run += f", {rn['dry']} run dry"
    else:
        run += "no round reaches you (they run dry or pass)"
    return text + run


# =============================================================================
# Order consistency: ships the prose names vs the structured target
# =============================================================================

def ships_named_in_text(text: str, names: Dict[str, str], candidates: Iterable[str],
                        ambiguous: Iterable[str] = ()) -> List[str]:
    """
    Ids of ``candidates`` (ship ids) that ``text`` names: by id, by full name,
    or by the name without its fleet prefix ('Lancer-1' for 'OCS Lancer-1')
    unless that short name is in ``ambiguous`` (e.g. an own ship's short name:
    both fleets field 'Lancer-1'). Word-bounded, case-insensitive, in order of
    first mention.
    """
    low = str(text or "").lower()
    skip = {a.lower() for a in ambiguous}
    hits = []
    for sid in candidates:
        name = str(names.get(sid) or "")
        forms = [sid.lower()]
        if name:
            forms.append(name.lower())
            short = name.split(" ", 1)[1].lower() if " " in name else ""
            if short and short not in skip:
                forms.append(short)
        best = None
        for form in forms:
            m = re.search(r"(?<![\w-])" + re.escape(form) + r"(?![\w-])", low)
            if m and (best is None or m.start() < best):
                best = m.start()
        if best is not None:
            hits.append((best, sid))
    return [sid for _, sid in sorted(hits)]


def order_text_target_note(args: Dict[str, Any], names: Dict[str, str], own_ids: Iterable[str],
                           enemy_ids: Iterable[str]) -> Optional[str]:
    """
    The warning for an issue_order whose prose names an enemy ship other
    than its structured target(s) (suggested_target / maneuver_target /
    torpedo_target): the engine and the compliance check apply the fields,
    the captain reads both - 11 of 25 recorded 'target deviations' were
    captains following the prose. None when they agree.
    """
    own = list(own_ids)
    short_own = [str(names.get(s, "")).split(" ", 1)[1] for s in own if " " in str(names.get(s, ""))]
    structured = []
    for key in ("suggested_target", "maneuver_target", "torpedo_target"):
        sid = resolve_ship_ref(args.get(key), names) if args.get(key) else None
        if sid:
            structured.append(sid)
    named = ships_named_in_text(args.get("order_text") or "", names, list(enemy_ids), short_own)
    extra = [sid for sid in named if sid not in structured]
    # Only the misleading cases get a tool result (and a follow-up call): the
    # prose names an enemy and there is no structured target, or the FIRST
    # enemy it names is not the structured target. A later "then shift to X"
    # is common (40% of the 1,049 recorded orders named a second enemy) and is
    # left to the captain's ORDERS block, which says which target flies.
    if not extra or (structured and named and named[0] in structured):
        return None
    lbl = lambda sid: label(names.get(sid), sid)  # noqa: E731
    if structured:
        applied = lbl(structured[0])
        return (f"order_text names {', '.join(lbl(s) for s in extra)} but the structured target is "
                f"{applied}: the engine and the compliance check apply {applied}; the captain reads "
                "both. If the text is what you mean, issue the order again with that target in "
                "suggested_target; otherwise say nothing.")
    return (f"order_text names {', '.join(lbl(s) for s in extra)} but the order has no structured "
            "target (suggested_target): the engine and the compliance check apply none, the captain "
            "picks from the text. Issue it again with suggested_target to make it the target; "
            "otherwise say nothing.")
