"""
Fleet-level doctrine for the free (rule-based) commanders.

HeuristicCaptain's "tactical" doctrine asks this module one question per
checkpoint - "what is my fleet doing?" - and flies its own ship inside that
answer. The answer (a FleetPlan) is computed once per faction per sim time
from the battle state alone and shared by every captain of that faction, so
it does not depend on the order (or thread) in which the captains decide.

The doctrine, in four behaviours:

1. Focus fire. The fleet picks ONE target - close to the fleet's centre,
   already damaged, thinly armoured, slow enough to hit (hittability) - and keeps it
   until it is dying or dead, or a markedly better one appears
   (FOCUS_SWITCH_MARGIN); torpedo salvos go at it first and gun ships with
   nothing in reach close on it. In reach, each gun ship weighs the same
   priorities from its own position (gun_target): guns only fire at their
   own ship's target, so a far focus would starve them.
2. Range discipline by hull role, nose first. Gun ships sprint in while a
   stop short of the enemy's guns stays possible (a closure cap), then coast
   nose-on through a controlled pass facing the nearest enemy gun and
   re-close nose first once the range opens; they do not brake into the
   enemy's guns (that shows the thin tail). They still pass through the
   enemy fleet - slower than the legacy charge - and HOLD_RANGE at their
   engagement range only when the closure happens to be nearly gone inside
   it (which can mean backing out to it). Torpedo boats HOLD_RANGE at a
   standoff outside the gun envelope.
3. Torpedo employment. Every boat fires every round it can launch before
   the next checkpoint at a target in reach (holding fire gains nothing:
   reloads do not stack), with the first round of every tube leaving in the
   same instant. Targets are ranked by torpedo_priority - big, slow hulls
   under a thin point-defence umbrella, close to the boats - not by the gun
   focus; rounds go first to the best target up to what saturates its PD,
   the rest to a second target if that one can be saturated too, else to
   the first.
4. Self-preservation and formation. The evade overlay runs from (or presents
   armour to) guided torpedoes; a crippled ship opens range; on the approach
   every gun ship matches the slowest gun hull's acceleration so the fleet
   arrives together under mutual PD cover. The flag ship announces focus and
   salvos on the FLEET NET (fleet_signal TARGET_CLAIM / TOT_SYNC).

Pure functions of the simulation state, no RNG.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Named thresholds
# ---------------------------------------------------------------------------

# Engagement (hold) ranges (km), from the doctrine's per-round P(hit) table
# vs a dodging destroyer (src/llm/prompts.py): spinal 82% / turret 73% at
# 50 km, 55% / 26% at 100 km - and a dodger the rest of the fleet keeps
# jinking (what focus fire does to it) 74% / 37% at 50 km but 21% / 3% at
# 100 km. A spinal hull holds at 60 km, a turret-only hull at 45 km.
SPINAL_ENGAGE_RANGE_KM = 60.0
TURRET_ENGAGE_RANGE_KM = 45.0
# Sprint: while the ship could still stop short of GUNS_HOT_KM after another
# full decision interval of burning (solo, at SPRINT_BRAKE_FRACTION of its
# drive, after a SPRINT_FLIP_S flip), it flies plain INTERCEPT, up to
# SPRINT_MAX_CLOSING_KPS; past that it coasts.
SPRINT_MAX_CLOSING_KPS = 4.0
# Capital ships (drive <= CAPITAL_ACCEL_G) turn too slowly to fight through a
# fast pass (a dreadnought needs ~100 s to turn its nose round after it, tail
# to the enemy meanwhile - round 1's dreadnoughts took 43 tail hits to 2 on
# the nose): they close at most at CAPITAL_MAX_CLOSING_KPS and start shooting
# from long range instead.
CAPITAL_ACCEL_G = 1.05
CAPITAL_MAX_CLOSING_KPS = 1.0
SPRINT_BRAKE_FRACTION = 0.7
SPRINT_FLIP_S = 40.0
# Inside engagement range x this, with the closure nearly gone, HOLD_RANGE.
HOLD_BAND_FACTOR = 1.35
HOLD_CLOSING_KPS = 0.4
# Nose first. Nose armour is ~5x the lateral/tail armour, and a ship braking
# against the closure (or opening range) turns its tail to the enemy. So a gun
# ship never brakes into a fight: beyond GUNS_HOT_KM it builds closure only
# while it could still stop short of GUNS_HOT_KM (sprint_ok - a closure cap,
# ~2 km/s), then coasts; inside it it coasts nose-on (PADLOCK: spinal and
# nose armour on him) through a controlled pass - every burn inside the gun
# envelope swings the nose off - and whenever it is outside the hold band
# with less than MIN_ADVANCE_KPS of closure (opening after a pass, or two
# fleets at rest) it burns toward him, nose first again: round 1 left two
# destroyers 220 km apart parked for a whole battle. Measured on mirror fleets vs the legacy captain: braking to a
# pass speed outside the guns (flip, tail toward a charging enemy at
# 200-300 km) lost 4 of 4 gun-heavy seeds; coasting in won 3 of 4.
GUNS_HOT_KM = 200.0
# A gun ship keeps facing the enemy gun it faces until another is this many
# times closer.
FACE_SWITCH_FACTOR = 1.5
MIN_ADVANCE_KPS = 0.5

# Torpedo boats (no guns) hold this range to the nearest enemy gun ship:
# outside the gun envelope that matters, inside the torpedoes' reach.
TORPEDO_STANDOFF_KM = 300.0
# A boat launches only at targets within this range (a longer flight gives
# the target's PD and evasion more time).
TORPEDO_LAUNCH_RANGE_KM = 600.0
# ...and fires whatever it has at a gun ship that got this close anyway.
TORPEDO_PANIC_RANGE_KM = 170.0
# Extra rounds on top of the computed saturating salvo (seeker losses).
SALVO_MARGIN_ROUNDS = 1
# Leftover tube capacity may open a second salvo on another target.
MAX_SALVOS_PER_CHECKPOINT = 2
# torpedo_priority scales: hull mass (a cruiser = 1), rounds needed to
# saturate, boat-to-target range.
TORPEDO_SIZE_REF_TONS = 3790.0
TORPEDO_NEED_REF_ROUNDS = 3.0
TORPEDO_RANGE_SCALE_KM = 300.0

# Focus selection.
FOCUS_RANGE_SCALE_KM = 100.0     # proximity weight halves at this range
FOCUS_DAMAGE_WEIGHT = 1.5        # x(1 + w * hull lost)
FOCUS_BREACH_BONUS = 0.5         # + when an armour facing is gone
FOCUS_ARMOR_REF_CM = 150.0       # thinner nose armour = easier kill
FOCUS_TORPEDO_BOAT_BONUS = 1.3   # its salvos kill our ships
# Agile hulls jink off gun rounds and outrun torpedoes: a target is worth
# 1 / (1 + its drive in g / FOCUS_AGILITY_REF_G) (a 3 g corvette 0.33, a 1 g
# battleship 0.6). Round 1 chased kiting corvettes with guns and torpedoes
# alike and landed 1-2 torpedoes a battle to legacy's 5-6 on big hulls.
FOCUS_AGILITY_REF_G = 1.5
FOCUS_SWITCH_MARGIN = 1.6        # a new focus must beat the old one by this
# Each gun ship weighs the same priorities (damage, breach, thin armour,
# torpedo boat) from ITS position, over a shorter range scale, with the fleet
# focus counted FOCUS_SHARE_BONUS times (a gun only fires at its own target).
# Measured on the merged pass-5 engine, gun-fleet mirrors vs legacy: a 2.0
# focus bonus won 0 of 4 (1 draw), 1.0 won 3 of 4 (1 draw) - in a gun brawl
# the fleets already converge on the damaged, soft targets this way, and a
# bonus only pulls fire off the ships that are shooting at you. The fleet
# focus still steers ships with nothing in reach and the torpedo salvos.
SHIP_RANGE_SCALE_KM = 80.0
FOCUS_SHARE_BONUS = 1.0
# "In reach": where the FIRE_WHEN_OPTIMAL gate still opens on a hull holding
# course (legacy capital ships fire from ~200 km on average).
GUN_REACH_KM = 250.0

# Self-preservation.
CRIPPLED_HULL_PCT = 35.0          # below this hull integrity: open range
RETREAT_RANGE_KM = 350.0          # ...to this range from the nearest gun ship
WITHDRAW_ANGLE_DEG = 60.0         # withdrawal burn off the line to the gun

# Battle damage (src/battle_damage.py). A hull whose drive makes less than
# FORMATION_MIN_DRIVE_FRACTION of its thrust no longer sets the fleet's
# approach speed (it would stall everyone); an enemy that has nothing left
# to shoot with is worth DISARMED_TARGET_FACTOR of a fighting one; a ship
# inside BLAST_AVOID_FACTOR x a dying hulk's reactor-blast radius burns away
# from it before it detonates.
FORMATION_MIN_DRIVE_FRACTION = 0.5
DISARMED_TARGET_FACTOR = 0.3
BLAST_AVOID_FACTOR = 1.5

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
    """'gun' (a working coilgun), 'torpedo' (loaded tubes only) or 'unarmed'
    (every gun mount wrecked and no round left in a working tube: battle
    damage, src/battle_damage.py, or an empty magazine)."""
    if has_guns(ship):
        return "gun"
    if has_torpedoes(ship):
        return "torpedo"
    return "unarmed"


def withdraw_direction(ship: Any, threat: Any) -> Any:
    """
    Burn direction for a withdrawal: away from ``threat`` but WITHDRAW_ANGLE_DEG
    off the line, so his guns see the flank instead of the tail (30% of tail
    hits strike the drive, src/battle_damage.py - a ship fleeing straight
    away had its drive shot out within two minutes). The side is the one the
    ship is already drifting to (deterministic tie-break: +z, then +y).
    """
    from ..physics import Vector3D
    away = ship.position - threat.position
    if away.magnitude < 1.0:
        away = Vector3D(-1.0, 0.0, 0.0)
    away = away.normalized()
    rel_v = ship.velocity - threat.velocity
    side = rel_v - away * rel_v.dot(away)
    if side.magnitude < 1.0:
        up = Vector3D(0.0, 0.0, 1.0) if abs(away.z) < 0.9 else Vector3D(0.0, 1.0, 0.0)
        side = up - away * up.dot(away)
    side = side.normalized()
    a = math.radians(WITHDRAW_ANGLE_DEG)
    return (away * math.cos(a) + side * math.sin(a)).normalized()


def drive_fraction(ship: Any) -> float:
    try:
        return float(ship.get_effective_thrust_fraction())
    except Exception:
        return 1.0


def blast_threat(simulation: Any, ship: Any) -> Optional[Any]:
    """The nearest dying hulk (either side) whose reactor blast would reach
    this ship with margin (BLAST_AVOID_FACTOR x its blast radius), or None.
    Hulks detonate 0-30 s after the killing blow."""
    from ..battle_damage import _hull_mass_tons, reactor_blast_radius_km
    worst = None
    for other in getattr(simulation, "ships", {}).values():
        if other is ship or not getattr(other, "is_dying", False) or other.is_destroyed:
            continue
        reach = BLAST_AVOID_FACTOR * reactor_blast_radius_km(_hull_mass_tons(other))
        d = _km(ship, other)
        if d < reach and (worst is None or d < worst[0]):
            worst = (d, other)
    return worst[1] if worst else None


def engage_range_km(ship: Any) -> float:
    return SPINAL_ENGAGE_RANGE_KM if has_spinal(ship) else TURRET_ENGAGE_RANGE_KM


def sprint_ok(range_km: float, engage_km: float, closing_kps: float, accel_ms2: float,
              interval_s: float = 30.0, max_closing_kps: float = SPRINT_MAX_CLOSING_KPS) -> bool:
    """
    True while a plain INTERCEPT burn for one more decision interval still
    leaves room to stop at ``engage_km`` (gun_maneuver passes GUNS_HOT_KM;
    see SPRINT_*).
    """
    if accel_ms2 <= 0.0 or closing_kps >= max_closing_kps:
        return False
    v0 = max(0.0, closing_kps) * 1000.0
    v1 = v0 + accel_ms2 * interval_s
    travel = v0 * interval_s + 0.5 * accel_ms2 * interval_s ** 2
    stop = v1 * v1 / (2.0 * SPRINT_BRAKE_FRACTION * accel_ms2) + v1 * SPRINT_FLIP_S
    return (range_km - engage_km) * 1000.0 - travel > stop


def gun_maneuver(range_km: float, closing_kps: float, engage_km: float, accel_ms2: float,
                 interval_s: float = 30.0,
                 max_closing_kps: float = SPRINT_MAX_CLOSING_KPS) -> Tuple[str, Dict[str, float]]:
    """
    (maneuver type, set_maneuver extras) for a gun ship vs its target: sprint
    while a stop short of his guns stays possible, coast nose-on through the
    pass, re-close nose first, hold range once the closure is gone (see
    GUNS_HOT_KM).
    """
    if (range_km <= engage_km * HOLD_BAND_FACTOR and abs(closing_kps) <= HOLD_CLOSING_KPS):
        return "HOLD_RANGE", {"range_km": engage_km}
    if closing_kps < MIN_ADVANCE_KPS and range_km > engage_km * HOLD_BAND_FACTOR:
        # Out of the hold band with (almost) no closure, or opening after a
        # pass: close, nose first. Never sit still out there.
        return "INTERCEPT", {}
    if range_km > GUNS_HOT_KM:
        if sprint_ok(range_km, GUNS_HOT_KM, closing_kps, accel_ms2, interval_s, max_closing_kps):
            return "INTERCEPT", {}
    return "PADLOCK", {}


def is_capital(ship: Any) -> bool:
    try:
        full = ship.kinematic_state.max_acceleration_ms2()
    except Exception:
        return False
    return 0.0 < full <= CAPITAL_ACCEL_G * STANDARD_G


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

def focus_priority(enemy: Any, centre: Any, range_scale_km: float = FOCUS_RANGE_SCALE_KM) -> float:
    """
    How much the fleet wants this enemy dead right now: close to the fleet's
    centre, already damaged or breached, thinly armoured, a torpedo boat.
    ``centre`` is a position (Vector3D).
    """
    d_km = (enemy.position - centre).magnitude / 1000.0
    proximity = 1.0 / (1.0 + d_km / range_scale_km)
    nose_cm, breached = _armor_cm(enemy)
    damage = 1.0 + FOCUS_DAMAGE_WEIGHT * (1.0 - hull_pct(enemy) / 100.0)
    if breached:
        damage += FOCUS_BREACH_BONUS
    softness = 1.0 / (1.0 + nose_cm / FOCUS_ARMOR_REF_CM)
    kind = role(enemy)
    bonus = (FOCUS_TORPEDO_BOAT_BONUS if kind == "torpedo" else
             DISARMED_TARGET_FACTOR if kind == "unarmed" else 1.0)
    return proximity * damage * softness * bonus * hittability(enemy)


def hittability(enemy: Any) -> float:
    return 1.0 / (1.0 + max_accel_ms2(enemy) / STANDARD_G / FOCUS_AGILITY_REF_G)


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


def gun_target(ship: Any, enemies: List[Any], focus_id: Optional[str]) -> Optional[Any]:
    """
    A gun ship's own target. Guns fire only at their ship's target, so a
    focus out of this ship's reach (GUN_REACH_KM) must not starve its guns:
    among the enemies in reach it takes the focus unless a much better one is
    on top of it (SHIP_RANGE_SCALE_KM, FOCUS_SHARE_BONUS); with none in reach
    it closes on the focus.
    """
    live = [e for e in enemies if _alive(e)]
    if not live:
        return None
    # A disarmed enemy out of reach cannot hurt us and is running: leave it
    # while anything armed is left (chasing it stalled battles to the limit).
    armed = [e for e in live if role(e) != "unarmed"]
    if armed:
        live = [e for e in live if role(e) != "unarmed" or _km(ship, e) <= GUN_REACH_KM]
    in_reach = [e for e in live if _km(ship, e) <= GUN_REACH_KM]
    if not in_reach:
        focus = next((e for e in live if e.ship_id == focus_id), None)
        return focus or min(live, key=lambda e: (_km(ship, e), e.ship_id))
    return max(in_reach, key=lambda e: (
        focus_priority(e, ship.position, SHIP_RANGE_SCALE_KM)
        * (FOCUS_SHARE_BONUS if e.ship_id == focus_id else 1.0), e.ship_id == focus_id,
        e.ship_id))


# ---------------------------------------------------------------------------
# Fleet plan
# ---------------------------------------------------------------------------

@dataclass
class Salvo:
    target_id: str
    rounds: Dict[str, int]       # boat id -> rounds this checkpoint
    need: int                    # rounds that saturate his point defence
    available: int               # rounds the boats in reach could launch
    saturating: bool = True      # False: a best-effort wave (see _salvo_plan)


@dataclass
class FleetPlan:
    faction: str
    time: float
    focus_id: Optional[str] = None
    # Torpedo salvos launched this checkpoint (empty = hold the rounds) and
    # each boat's share: boat id -> [(target id, rounds), ...].
    salvos: List[Salvo] = field(default_factory=list)
    launches: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
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


def torpedo_priority(enemy: Any, boats: List[Any], need: int) -> float:
    """
    Worth of a torpedo target: a big, slow hull (it cannot outrun or out-jink
    the round, and a kill is worth more) under a thin PD umbrella (few rounds
    needed to saturate it), close to the boats (short flight, brief PD
    dwell), already damaged.
    """
    from ..battle_damage import _hull_mass_tons
    size = math.sqrt(max(_hull_mass_tons(enemy), 1.0) / TORPEDO_SIZE_REF_TONS)
    d_km = min(_km(b, enemy) for b in boats)
    damage = 1.0 + FOCUS_DAMAGE_WEIGHT * (1.0 - hull_pct(enemy) / 100.0)
    return (size * damage * hittability(enemy) ** 2
            / (1.0 + need / TORPEDO_NEED_REF_ROUNDS) / (1.0 + d_km / TORPEDO_RANGE_SCALE_KM))


def _salvo_plan(simulation: Any, boats: List[Any], enemies: List[Any], focus: Optional[Any],
                fleet_data: Optional[Dict[str, Any]]) -> List[Salvo]:
    """
    This checkpoint's torpedo employment. Every boat with a round it can
    launch before the next checkpoint and a target in reach fires it - a
    tube that holds fire gains nothing (reloads do not stack). Targets by
    torpedo_priority; the first round of every tube leaves in the same
    instant (the rest follow at the tube's reload). Rounds go first to the
    best target up to what saturates its point defence (salvo_to_saturate
    over its own turrets plus escorts within PD range, + SALVO_MARGIN_ROUNDS,
    less rounds already inbound); what is left opens a second salvo when it
    can saturate that one too, else joins the first (MAX_SALVOS_PER_CHECKPOINT).
    A gun ship already inside TORPEDO_PANIC_RANGE_KM of a boat gets that
    boat's rounds first.

    Rounds that cannot saturate anything are launched anyway, on purpose:
    a few single-tube boats can never assemble a saturating salvo, and
    holding them for one (round 1) cost the battleship roster 4 of 4
    mirror battles against legacy, which lands 5-6 such rounds a battle on
    big hulls. The captain's launch check calls these TRICKLE (per launch
    call, so the boats of one synchronized salvo each register one too).
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

    def shooters(enemy) -> List[Any]:
        return sorted((b for b in boats if left[b.ship_id] > 0
                       and _km(b, enemy) <= TORPEDO_LAUNCH_RANGE_KM),
                      key=lambda b: (_km(b, enemy), b.ship_id))

    def take(enemy, want: Optional[int]) -> Dict[str, int]:
        rounds: Dict[str, int] = {}
        for b in shooters(enemy):
            if want is not None and want <= 0:
                break
            n = left[b.ship_id] if want is None else min(left[b.ship_id], want)
            rounds[b.ship_id] = n
            left[b.ship_id] -= n
            if want is not None:
                want -= n
        return rounds

    needs = {e.ship_id: need_for(e) for e in enemies}
    salvos: List[Salvo] = []
    # A gun ship on top of a boat: everything that boat has, first.
    for enemy in sorted(enemies, key=lambda e: e.ship_id):
        close = [b for b in boats if left[b.ship_id] > 0 and has_guns(enemy)
                 and _km(b, enemy) <= TORPEDO_PANIC_RANGE_KM]
        if close:
            rounds = {}
            for b in close:
                rounds[b.ship_id] = left[b.ship_id]
                left[b.ship_id] = 0
            salvos.append(Salvo(enemy.ship_id, rounds, needs[enemy.ship_id],
                                sum(rounds.values()),
                                sum(rounds.values()) >= needs[enemy.ship_id]))
    ranked = sorted((e for e in enemies if needs[e.ship_id] > 0 and shooters(e)),
                    key=lambda e: (-torpedo_priority(e, boats, needs[e.ship_id]), e.ship_id))
    for enemy in ranked:
        if len(salvos) >= MAX_SALVOS_PER_CHECKPOINT or not shooters(enemy):
            continue
        need = needs[enemy.ship_id]
        available = sum(left[b.ship_id] for b in shooters(enemy))
        if salvos and available < need:
            continue        # a second target only if it can be saturated too
        rounds = take(enemy, need if available >= need else None)
        salvos.append(Salvo(enemy.ship_id, rounds, need, available, available >= need))
    # Rounds still in the tubes join the first salvo they can reach.
    for salvo in salvos:
        target = next((e for e in enemies if e.ship_id == salvo.target_id), None)
        if target is None:
            continue
        for boat_id, n in take(target, None).items():
            salvo.rounds[boat_id] = salvo.rounds.get(boat_id, 0) + n
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
            if n > 0:
                plan.launches.setdefault(boat_id, []).append((salvo.target_id, n))
    plan.torpedo_target_id = plan.salvos[0].target_id if plan.salvos else plan.focus_id

    gunships = [s for s in friends if role(s) == "gun"]
    accels = [max_accel_ms2(s) for s in gunships
              if drive_fraction(s) >= FORMATION_MIN_DRIVE_FRACTION and max_accel_ms2(s) > 0]
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
