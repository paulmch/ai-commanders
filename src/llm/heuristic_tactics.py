"""
Fleet-level doctrine for the free (rule-based) commanders.

HeuristicCaptain's "tactical" doctrine asks this module one question per
checkpoint - "what is my fleet doing?" - and flies its own ship inside that
answer. The answer (a FleetPlan) is computed once per faction per sim time
from the battle state alone and shared by every captain of that faction, so
it does not depend on the order (or thread) in which the captains decide.

The doctrine, in four behaviours:

1. Focus fire. The fleet picks ONE gun target - the enemy that is close to
   the fleet's centre, already damaged, thinly armoured or a soft torpedo
   boat - and keeps it until it is dying or dead, or a markedly better
   target appears (FOCUS_SWITCH_MARGIN). Every gun ship shoots it unless an
   enemy is right on top of that ship.
2. Range discipline by hull role. Gun ships close to their engagement range
   (INTERCEPT with standoff_km + a slow arrival) and then HOLD_RANGE there
   instead of merging; torpedo boats hold a standoff outside the enemy's
   effective gun range.
3. Massed torpedo salvos. Torpedo boats hold their rounds until the fleet
   can put enough into the air at once to saturate the target's point
   defence (its own turrets plus escorts within PD range), then every boat
   launches at the same checkpoint, so the rounds arrive together.
4. Self-preservation and formation. The evade overlay dodges rounds when
   enemy guns are in reach (and runs from torpedoes); a crippled ship opens
   range; on the approach every gun ship matches the slowest gun hull's
   acceleration so the fleet arrives together under mutual PD cover.

Pure functions of the simulation state, no RNG.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Named thresholds
# ---------------------------------------------------------------------------

# Engagement ranges (km), from the doctrine's per-round P(hit) table vs a
# dodging destroyer (src/llm/prompts.py): spinal 82% / turret 73% at 50 km,
# 55% / 26% at 100 km - and a dodger the rest of the fleet keeps jinking
# (what focus fire does to it) 74% / 37% at 50 km but 21% / 3% at 100 km.
# Focus fire is fought where the jinking target is still hit: a spinal hull
# holds at 60 km, a turret-only hull closes to 45 km.
SPINAL_ENGAGE_RANGE_KM = 60.0
TURRET_ENGAGE_RANGE_KM = 45.0
# INTERCEPT arrival closing speed (km/s) at the engagement range: slow
# enough to hold the range once there, fast enough not to dawdle.
ARRIVAL_CLOSING_KPS = 0.6
# Sprint: while the ship could still stop short of its engagement range
# after another full decision interval of burning (solo, at
# SPRINT_BRAKE_FRACTION of its drive, after a SPRINT_FLIP_S flip), it flies
# plain INTERCEPT, up to SPRINT_MAX_CLOSING_KPS; then the standoff INTERCEPT
# plans the brake. Both fleets range keeping on each other otherwise crawl in
# at ~1.5 km/s (the engine plans each stop around the other's charge).
SPRINT_MAX_CLOSING_KPS = 4.0
SPRINT_BRAKE_FRACTION = 0.7
SPRINT_FLIP_S = 40.0
# Inside engagement range x this the approach is over: HOLD_RANGE.
HOLD_BAND_FACTOR = 1.35

# Torpedo boats (no guns) hold this range to the nearest enemy gun ship:
# outside the gun envelope that matters, inside the torpedoes' reach.
TORPEDO_STANDOFF_KM = 300.0
# A boat launches only at targets within this range (a longer flight gives
# the target's PD and evasion more time).
TORPEDO_LAUNCH_RANGE_KM = 480.0
# ...and fires whatever it has at a gun ship that got this close anyway.
TORPEDO_PANIC_RANGE_KM = 170.0
# Extra rounds on top of the computed saturating salvo (seeker losses).
SALVO_MARGIN_ROUNDS = 1
# Leftover tube capacity may open a second salvo on another target.
MAX_SALVOS_PER_CHECKPOINT = 2

# Focus selection.
FOCUS_RANGE_SCALE_KM = 200.0     # proximity weight halves at this range
FOCUS_DAMAGE_WEIGHT = 1.5        # x(1 + w * hull lost)
FOCUS_BREACH_BONUS = 0.5         # + when an armour facing is gone
FOCUS_ARMOR_REF_CM = 150.0       # thinner nose armour = easier kill
FOCUS_TORPEDO_BOAT_BONUS = 1.3   # soft, and its salvos kill our ships
FOCUS_SWITCH_MARGIN = 1.6        # a new focus must beat the old one by this
# A gun ship shoots the enemy nearest to it instead of the focus when that
# enemy is inside its engagement range and the focus is this many times
# further away.
SELF_DEFENSE_FACTOR = 2.5

# Self-preservation.
CRIPPLED_HULL_PCT = 35.0          # below this hull integrity: open range
RETREAT_RANGE_KM = 350.0          # ...to this range from the nearest gun ship
GUN_THREAT_RANGE_KM = 260.0       # enemy guns this close: evade overlay 'dodge'

STANDARD_G = 9.81


# ---------------------------------------------------------------------------
# Ship roles
# ---------------------------------------------------------------------------

def _alive(ship: Any) -> bool:
    return not (getattr(ship, "is_destroyed", False) or getattr(ship, "is_dying", False)
                or getattr(ship, "is_surrendered", False))


def _gun_states(ship: Any) -> List[Any]:
    """Operational coilgun mounts (ship.weapons also holds torpedo tubes and
    PD turrets)."""
    out = []
    for ws in (getattr(ship, "weapons", None) or {}).values():
        weapon = getattr(ws, "weapon", None)
        if weapon is None or not getattr(ws, "is_operational", True):
            continue
        if "coil" in str(getattr(weapon, "weapon_type", "")).lower():
            out.append(ws)
    return out


def has_guns(ship: Any) -> bool:
    return bool(_gun_states(ship))


def has_spinal(ship: Any) -> bool:
    return any(not getattr(ws.weapon, "is_turreted", True) for ws in _gun_states(ship))


def has_torpedoes(ship: Any) -> bool:
    tubes = getattr(ship, "torpedo_launchers", None) or (
        [ship.torpedo_launcher] if getattr(ship, "torpedo_launcher", None) else [])
    return any(getattr(t, "current_magazine", 0) > 0 for t in tubes)


def role(ship: Any) -> str:
    """'gun' (any coilgun), 'torpedo' (launchers only) or 'unarmed'."""
    if has_guns(ship):
        return "gun"
    if getattr(ship, "torpedo_launcher", None) is not None or getattr(ship, "torpedo_launchers", None):
        return "torpedo"
    return "unarmed"


def engage_range_km(ship: Any) -> float:
    return SPINAL_ENGAGE_RANGE_KM if has_spinal(ship) else TURRET_ENGAGE_RANGE_KM


def sprint_ok(range_km: float, engage_km: float, closing_kps: float, accel_ms2: float,
              interval_s: float = 30.0) -> bool:
    """
    True while a plain INTERCEPT burn for one more decision interval still
    leaves room to stop at the engagement range (see SPRINT_*).
    """
    if accel_ms2 <= 0.0 or closing_kps >= SPRINT_MAX_CLOSING_KPS:
        return False
    v0 = max(0.0, closing_kps) * 1000.0
    v1 = v0 + accel_ms2 * interval_s
    travel = v0 * interval_s + 0.5 * accel_ms2 * interval_s ** 2
    stop = v1 * v1 / (2.0 * SPRINT_BRAKE_FRACTION * accel_ms2) + v1 * SPRINT_FLIP_S
    return (range_km - engage_km) * 1000.0 - travel > stop


def max_accel_ms2(ship: Any) -> float:
    try:
        return (ship.kinematic_state.max_acceleration_ms2()
                * ship.get_effective_thrust_fraction())
    except Exception:
        return 0.0


def hull_pct(ship: Any) -> float:
    try:
        return float(ship.hull_integrity)
    except Exception:
        return 100.0


def _armor_cm(ship: Any) -> Tuple[float, bool]:
    """(nose armour cm, any facing breached)."""
    armor = getattr(ship, "armor", None)
    sections = getattr(armor, "sections", None) or {}
    nose = 0.0
    breached = False
    for loc, sec in sections.items():
        cm = float(getattr(sec, "thickness_cm", 0.0) or 0.0)
        if cm <= 0.0:
            breached = True
        if "nose" in str(getattr(loc, "value", loc)).lower():
            nose = cm
    return nose, breached


def _km(a: Any, b: Any) -> float:
    return (a.position - b.position).magnitude / 1000.0


# ---------------------------------------------------------------------------
# Focus target
# ---------------------------------------------------------------------------

def focus_priority(enemy: Any, centre: Any) -> float:
    """
    How much the fleet wants this enemy dead right now: close to the fleet's
    centre, already damaged or breached, thinly armoured, a torpedo boat.
    ``centre`` is a position (Vector3D).
    """
    d_km = (enemy.position - centre).magnitude / 1000.0
    proximity = 1.0 / (1.0 + d_km / FOCUS_RANGE_SCALE_KM)
    nose_cm, breached = _armor_cm(enemy)
    damage = 1.0 + FOCUS_DAMAGE_WEIGHT * (1.0 - hull_pct(enemy) / 100.0)
    if breached:
        damage += FOCUS_BREACH_BONUS
    softness = 1.0 / (1.0 + nose_cm / FOCUS_ARMOR_REF_CM)
    bonus = FOCUS_TORPEDO_BOAT_BONUS if role(enemy) == "torpedo" else 1.0
    return proximity * damage * softness * bonus


def pick_focus(enemies: List[Any], centre: Any, previous_id: Optional[str]) -> Optional[Any]:
    """Best focus target, sticky: the previous one stays unless beaten by
    FOCUS_SWITCH_MARGIN. Ties break on ship id (deterministic)."""
    live = [e for e in enemies if _alive(e)]
    if not live:
        return None
    scored = sorted(((focus_priority(e, centre), e.ship_id, e) for e in live),
                    key=lambda t: (-t[0], t[1]))
    best_score, _, best = scored[0]
    for score, sid, enemy in scored:
        if sid == previous_id:
            if best_score <= score * FOCUS_SWITCH_MARGIN:
                return enemy
            break
    return best


# ---------------------------------------------------------------------------
# Fleet plan
# ---------------------------------------------------------------------------

@dataclass
class Salvo:
    target_id: str
    rounds: Dict[str, int]       # boat id -> rounds this checkpoint
    need: int                    # rounds that saturate his point defence
    available: int               # rounds the boats in reach could launch


@dataclass
class FleetPlan:
    faction: str
    time: float
    focus_id: Optional[str] = None
    # Torpedo salvos launched this checkpoint (empty = hold the rounds) and
    # each boat's share: boat id -> (target id, rounds).
    salvos: List[Salvo] = field(default_factory=list)
    launches: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    # The boats' target: the first salvo's, else the gun focus.
    torpedo_target_id: Optional[str] = None
    # Approach throttle per gun ship: matched to the slowest gun hull.
    formation_throttle: Dict[str, float] = field(default_factory=dict)
    # The ship that announces the plan on the FLEET NET (fleet_signal):
    # the first live gun ship (else the first live ship) by id.
    flag_id: Optional[str] = None
    focus_changed: bool = False


_LOCK = threading.Lock()
_STATE_ATTR = "_heuristic_fleet_tactics"


def _centre(ships: List[Any]) -> Any:
    pos = ships[0].position * 0.0
    for s in ships:
        pos = pos + s.position
    return pos * (1.0 / len(ships))


def _salvo_plan(simulation: Any, boats: List[Any], enemies: List[Any], focus: Optional[Any],
                fleet_data: Optional[Dict[str, Any]]) -> List[Salvo]:
    """
    This checkpoint's salvos. Each is sized to the target's point defence
    (salvo_to_saturate over its own turrets plus escorts within PD range,
    + SALVO_MARGIN_ROUNDS, less friendly rounds already inbound) and drawn
    from the boats nearest to it; capacity left over may take another target
    (at most MAX_SALVOS_PER_CHECKPOINT). Targets: the gun focus first, then
    the cheapest to saturate. A target the boats cannot saturate gets
    nothing (the rounds are held) unless it is a gun ship already inside
    TORPEDO_PANIC_RANGE_KM of a boat: that one gets everything in reach.
    """
    from .fire_solutions import pd_cover, rounds_inbound, salvo_to_saturate
    from .prompts import pd_doctrine_numbers

    if not boats or not enemies:
        return []
    pd = pd_doctrine_numbers(fleet_data)
    left = {b.ship_id: int(simulation.torpedo_launch_capacity(b.ship_id)) for b in boats}

    def need_for(enemy) -> int:
        own, escorts, _ = pd_cover(simulation, enemy, pd["range_km"])
        need = salvo_to_saturate(own + escorts, pd["turrets_to_blind_at_floor"])
        inbound = len(rounds_inbound(simulation, boats[0].faction, enemy.ship_id))
        return max(0, need + SALVO_MARGIN_ROUNDS - inbound)

    needs = {e.ship_id: need_for(e) for e in enemies}
    candidates = sorted(enemies, key=lambda e: (e is not focus, needs[e.ship_id], e.ship_id))
    salvos: List[Salvo] = []
    for enemy in candidates:
        if len(salvos) >= MAX_SALVOS_PER_CHECKPOINT:
            break
        shooters = sorted((b for b in boats if left[b.ship_id] > 0
                           and _km(b, enemy) <= TORPEDO_LAUNCH_RANGE_KM),
                          key=lambda b: (_km(b, enemy), b.ship_id))
        available = sum(left[b.ship_id] for b in shooters)
        need = needs[enemy.ship_id]
        if available <= 0:
            continue
        panic = has_guns(enemy) and any(_km(b, enemy) <= TORPEDO_PANIC_RANGE_KM for b in shooters)
        if panic:
            want = available
        elif need == 0 or available < need:
            continue          # enough already inbound / cannot saturate: hold
        else:
            want = need
        rounds: Dict[str, int] = {}
        for b in shooters:
            if want <= 0:
                break
            n = min(left[b.ship_id], want)
            rounds[b.ship_id] = n
            left[b.ship_id] -= n
            want -= n
        salvos.append(Salvo(target_id=enemy.ship_id, rounds=rounds, need=need,
                            available=available))
    return salvos


def compute_fleet_plan(simulation: Any, faction: str, previous: Optional[FleetPlan],
                       fleet_data: Optional[Dict[str, Any]] = None) -> FleetPlan:
    now = float(getattr(simulation, "current_time", 0.0))
    plan = FleetPlan(faction=faction, time=now)
    ships = [s for s in getattr(simulation, "ships", {}).values() if _alive(s)]
    friends = sorted((s for s in ships if s.faction == faction), key=lambda s: s.ship_id)
    enemies = sorted((s for s in ships if s.faction != faction), key=lambda s: s.ship_id)
    if not friends or not enemies:
        return plan

    centre = _centre(friends)
    focus = pick_focus(enemies, centre, previous.focus_id if previous else None)
    plan.focus_id = focus.ship_id if focus else None
    plan.focus_changed = plan.focus_id != (previous.focus_id if previous else None)
    plan.flag_id = next((s.ship_id for s in friends if role(s) == "gun"), friends[0].ship_id)

    boats = [s for s in friends if role(s) == "torpedo" and has_torpedoes(s)]
    plan.salvos = _salvo_plan(simulation, boats, enemies, focus, fleet_data)
    for salvo in plan.salvos:
        for boat_id, n in salvo.rounds.items():
            plan.launches[boat_id] = (salvo.target_id, n)
    plan.torpedo_target_id = plan.salvos[0].target_id if plan.salvos else plan.focus_id

    gunships = [s for s in friends if role(s) == "gun"]
    accels = [a for a in (max_accel_ms2(s) for s in gunships) if a > 0]
    if accels:
        slowest = min(accels)
        for s in gunships:
            a = max_accel_ms2(s)
            plan.formation_throttle[s.ship_id] = round(min(1.0, slowest / a), 3) if a > 0 else 1.0
    return plan


def fleet_plan(simulation: Any, faction: str,
               fleet_data: Optional[Dict[str, Any]] = None) -> FleetPlan:
    """The faction's plan for the current sim time (computed once, shared)."""
    now = float(getattr(simulation, "current_time", 0.0))
    with _LOCK:
        store = getattr(simulation, _STATE_ATTR, None)
        if store is None:
            store = {}
            setattr(simulation, _STATE_ATTR, store)
        current = store.get(faction)
        if current is not None and current.time == now:
            return current
        plan = compute_fleet_plan(simulation, faction, current, fleet_data)
        store[faction] = plan
        return plan
