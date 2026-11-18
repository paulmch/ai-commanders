"""
Experimental doctrine variants for the free commanders: the static "wall"
the LLM fleets fly, and pincer / envelopment / two-axis torpedo strikes
against it. Behind a switch (HeuristicCaptain doctrine), never the default.

Doctrines (FLANK_DOCTRINES):

- wall: every gun ship closes slowly (closure <= WALL_CLOSE_KPS) to
  WALL_ENGAGE_KM of the nearest enemy gun, then coasts PADLOCKed nose-on to
  the nearest armed enemy - the wall's natural counter to a flanker: the
  ships nearest a flanking group face it - and never burns again. Guns pick
  targets as the tactical doctrine does (fleet focus, nearby soft hulls
  first: thin flankers that come in reach get shot). Torpedo boats as in
  the tactical doctrine.
- pincerNN: the fast gun hulls (full drive >= FLANKER_MIN_ACCEL_G) fly wide
  to a station NN degrees off the enemy's facing (the axis from the enemy
  centre to our holding group) at POSITION_RANGE_KM - outside the enemy's
  effective gun range against a maneuvering small hull - with the evade
  overlay on, then all turn in together on a committed attack run to their
  own engagement range. The heavy hulls hold as a wall at HOLD_WAIT_KM until
  the attack run starts (or FLANK_TIMEOUT_S), then close like the wall.
- pincerNNc: the same, but the station is at the attack range itself: the
  flankers cut the corner through the enemy's gun envelope.
- envelopNN: no centre: the gun ships split into two groups (alternating by
  drive, so each has fast and slow hulls) that position at +NN and -NN.
- split: the tactical doctrine, but torpedo boats take station on two
  bearings SPLIT_HALF_DEG either side of the axis, so their (simultaneous)
  salvos arrive from directions 2 x SPLIT_HALF_DEG apart.

Pure functions of the battle state plus a per-faction memo of the group
assignment (fixed at the first checkpoint) and of the attack trigger.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import heuristic_tactics as tactics

# Fast hulls fly the flank (frigates 3 g, destroyers 2 g); slower hulls hold.
FLANKER_MIN_ACCEL_G = 1.9
# Effective range of the wall's guns against a small hull that keeps
# maneuvering: the calibrated forecast for a dodging frigate is 19 % at
# 100 km and 4 % at 150 km (rounds fly 30-50 s at 300 km while a 3 g hull
# displaces ~13 km in 30 s against a 500 m gate). Positioning happens
# beyond POSITION_RANGE_KM from the enemy centre, with the evade overlay.
POSITION_RANGE_KM = 250.0
# Cut-corner variant: the station is at this range (inside the envelope).
CUT_CORNER_RANGE_KM = 110.0
STATION_TOL_KM = 35.0
STATION_TOL_DEG = 12.0
# On station: within this band of the station range (km below, above).
STATION_RANGE_BAND_KM = (60.0, 100.0)
# The attack run starts once this share of the live flankers is on station.
ATTACK_QUORUM = 0.75
# The heavy (holding) group waits at this range from the nearest enemy gun
# until the flank attack starts, or FLANK_TIMEOUT_S has passed.
HOLD_WAIT_KM = 230.0
FLANK_TIMEOUT_S = 330.0
# Wall: close no faster than this to WALL_ENGAGE_KM, then coast PADLOCKed.
WALL_CLOSE_KPS = 0.8
WALL_ENGAGE_KM = 140.0
# Two-axis torpedo strike: boats on bearings +-SPLIT_HALF_DEG off the axis.
SPLIT_HALF_DEG = 35.0


@dataclass(frozen=True)
class FlankSpec:
    kind: str                      # 'wall' | 'pincer' | 'envelop' | 'split'
    angle_deg: float = 0.0
    station_km: float = POSITION_RANGE_KM
    split_boats: bool = False      # torpedo boats strike from two bearings


FLANK_DOCTRINES: Dict[str, FlankSpec] = {
    "wall": FlankSpec("wall"),
    "pincer45": FlankSpec("pincer", 45.0),
    "pincer70": FlankSpec("pincer", 70.0),
    "pincer90": FlankSpec("pincer", 90.0),
    "pincer70c": FlankSpec("pincer", 70.0, CUT_CORNER_RANGE_KM),
    "envelop70": FlankSpec("envelop", 70.0),
    "split": FlankSpec("split"),
}


# ---------------------------------------------------------------------------
# The default doctrine's switch rule ("tactical")
# ---------------------------------------------------------------------------
# Re-derived on the current engine (scripts/flank_experiment.py, tuning seeds
# 1-4, mirror rosters, both sides; docs/pass-5-findings.md "Flank experiment"):
#
# - Every cut-corner pincer (45/70/90 deg at 80/110/150 km) beat the wall
#   16/0/0 on the dreadnought rosters (flagship, dread_fast) and was mixed
#   or lost on battleship lines (battleline, combined, escort) and on fast
#   lines (gunline, core_fast). The deciding enemy property is a gun line
#   whose power sits on dreadnought-class hulls (full drive <= 0.9 g, 50 s per
#   90 deg): PINCER_ENEMY_ACCEL_G / PINCER_CAPITAL_SHARE, with dread_fast at
#   150 points (0.40 of its gun energy on its dreadnought) still 8/0/0.
#   Two flankers were enough (flagship at 150 points: 2 frigates, 8/0/0).
# - Station: 70 deg at 150 km killed 100 for 23 losses over 32 battles on
#   those rosters (110 km: 93-48, 80 km: 86-51, 250 km outside the guns:
#   55-17 with far fewer kills; 90 deg at 150 km 100-26, 45 deg 82-38).
# - Against a BATTLESHIP line (>= HEAVY_LINE_SHARE of the enemy's gun energy
#   on hulls <= HEAVY_ENEMY_ACCEL_G that are not dreadnought class) the
#   previous default ("classic": close nose-first and pass) beat the wall
#   (battleline 12/1/3, combined 14/2/0, escort 9/4/3); the wall beat it on
#   fast lines (gunline 10/1/5, core_fast 8/5/3).
CAPITAL_ACCEL_G = 1.05            # (kept for the experiment's features)
PINCER_ENEMY_ACCEL_G = 0.9
PINCER_CAPITAL_SHARE = 0.3
PINCER_MIN_FAST_HULLS = 2
PINCER_ANGLE_DEG = 70.0
PINCER_STATION_KM = 150.0
HEAVY_ENEMY_ACCEL_G = 1.05
HEAVY_LINE_SHARE = 0.5
# A gun fleet facing a TORPEDO fleet (>= TORPEDO_FLEET_SHARE of its hulls
# torpedo-only) must close fast and kill the boats: a wall closing at
# 0.8 km/s lost 2/1/21 to legacy on held-out seeds where classic won 14/2/8.
TORPEDO_FLEET_SHARE = 0.5
# The two-axis torpedo strike no longer paid on the current engine (torpedo
# datalink, sprint profile): split v classic 4/0/12 on the torpedo roster.
DEFAULT_SPLIT_BOATS = False
WALL_SPEC = FlankSpec("wall", split_boats=DEFAULT_SPLIT_BOATS)
PINCER_SPEC = FlankSpec("pincer", PINCER_ANGLE_DEG, PINCER_STATION_KM, split_boats=DEFAULT_SPLIT_BOATS)
CLASSIC_SPEC = None               # the classic doctrine (heuristic_tactics only)


def _gun_energy(ship: Any) -> float:
    energy = tactics._mount_energy_gj_s(getattr(ship, "ship_type", ""))
    return sum(v for slot, v in energy.items() if not str(slot).startswith("pd"))


def energy_share_at_or_below(ships: List[Any], accel_g: float) -> float:
    """Share of these ships' gun energy (GJ/s) on hulls whose full drive is
    at or below ``accel_g``."""
    total = part = 0.0
    for s in ships:
        e = _gun_energy(s)
        total += e
        if 0.0 < full_accel_g(s) <= accel_g:
            part += e
    return part / total if total > 0 else 0.0


def capital_energy_share(ships: List[Any]) -> float:
    return energy_share_at_or_below(ships, CAPITAL_ACCEL_G)


def torpedo_hull_share(ships: List[Any]) -> float:
    """Share of these hulls that fight with torpedoes only."""
    return (sum(1 for s in ships if tactics.role(s) == "torpedo") / len(ships)) if ships else 0.0


def fast_hull_count(ships: List[Any]) -> int:
    return sum(1 for s in ships if tactics.role(s) == "gun" and full_accel_g(s) >= FLANKER_MIN_ACCEL_G)


def choose_default_spec(friends: List[Any], enemies: List[Any]) -> Optional[FlankSpec]:
    """The default doctrine's plan from the two compositions alone (so a
    mirror battle chooses the same plan on both sides): the classic
    close-and-pass against a torpedo fleet (when we are not one) and against a
    battleship line, the pincer against a dreadnought line when we have the
    flankers, else the wall. None = classic."""
    if (torpedo_hull_share(enemies) >= TORPEDO_FLEET_SHARE
            and torpedo_hull_share(friends) < TORPEDO_FLEET_SHARE):
        return CLASSIC_SPEC
    if (energy_share_at_or_below(enemies, PINCER_ENEMY_ACCEL_G) >= PINCER_CAPITAL_SHARE
            and fast_hull_count(friends) >= PINCER_MIN_FAST_HULLS):
        return PINCER_SPEC
    if energy_share_at_or_below(enemies, HEAVY_ENEMY_ACCEL_G) >= HEAVY_LINE_SHARE:
        return CLASSIC_SPEC
    return WALL_SPEC


_DEFAULT_ATTR = "_heuristic_default_spec"


def default_spec(simulation: Any, faction: str) -> Optional[FlankSpec]:
    """The plan the default doctrine flies for this faction, chosen at its
    first checkpoint from the fleets as they started and then kept
    (None = classic)."""
    with _LOCK:
        store = getattr(simulation, _DEFAULT_ATTR, None)
        if store is None:
            store = {}
            setattr(simulation, _DEFAULT_ATTR, store)
        if faction not in store:
            ships = [s for s in simulation.ships.values() if _alive(s)]
            store[faction] = choose_default_spec([s for s in ships if s.faction == faction],
                                                 [s for s in ships if s.faction != faction])
        return store[faction]


_SWEEP = None


def flank_spec(name: Optional[str]) -> Optional[FlankSpec]:
    """The FlankSpec of a named doctrine, or of a parameter-sweep name:
    'pincer{angle}c{range_km}' / 'pincer{angle}r{range_km}' (station range)
    and 'envelop{angle}' - used by the experiment runner to re-derive the
    thresholds; None for a non-flank doctrine."""
    global _SWEEP
    if not name:
        return None
    if name in FLANK_DOCTRINES:
        return FLANK_DOCTRINES[name]
    import re
    if _SWEEP is None:
        _SWEEP = (re.compile(r"^pincer(\d+)[cr](\d+)$"), re.compile(r"^envelop(\d+)$"))
    m = _SWEEP[0].match(name)
    if m:
        return FlankSpec("pincer", float(m.group(1)), float(m.group(2)))
    m = _SWEEP[1].match(name)
    if m:
        return FlankSpec("envelop", float(m.group(1)))
    return None


def _alive(s: Any) -> bool:
    return tactics._alive(s)


def full_accel_g(ship: Any) -> float:
    try:
        return ship.kinematic_state.max_acceleration_ms2() / tactics.STANDARD_G
    except Exception:
        return 0.0


def assign_groups(ships: List[Any], spec: FlankSpec) -> Dict[str, str]:
    """ship id -> 'hold' | 'flank_a' | 'flank_b' | 'boat' (deterministic,
    from hull classes only, so analysis can recompute it from a roster)."""
    out: Dict[str, str] = {}
    guns = sorted((s for s in ships if tactics.role(s) == "gun"),
                  key=lambda s: (-full_accel_g(s), s.ship_id))
    for s in ships:
        if tactics.role(s) != "gun":
            out[s.ship_id] = "boat"
    if spec.kind == "pincer":
        fast = [s for s in guns if full_accel_g(s) >= FLANKER_MIN_ACCEL_G]
        slow = [s for s in guns if full_accel_g(s) < FLANKER_MIN_ACCEL_G]
        if not slow and fast:            # all fast: the slowest half holds
            k = max(1, len(fast) // 2)
            slow, fast = fast[-k:], fast[:-k]
        for s in fast:
            out[s.ship_id] = "flank_a"
        for s in slow:
            out[s.ship_id] = "hold"
    elif spec.kind == "envelop":
        for i, s in enumerate(guns):
            out[s.ship_id] = "flank_a" if i % 2 == 0 else "flank_b"
    else:
        for s in guns:
            out[s.ship_id] = "hold"
    return out


@dataclass
class FlankState:
    groups: Dict[str, str] = field(default_factory=dict)
    attack_started: bool = False
    attack_t: Optional[float] = None
    time: float = -1.0
    stations: Dict[str, Any] = field(default_factory=dict)   # group -> station (Vector3D)
    axis: Any = None


_LOCK = threading.Lock()
_ATTR = "_heuristic_flank_state"


def _centre(ships: List[Any]) -> Any:
    pos = ships[0].position * 0.0
    for s in ships:
        pos = pos + s.position
    return pos * (1.0 / len(ships))


def _rot_z(v: Any, deg: float) -> Any:
    from ..physics import Vector3D
    a = math.radians(deg)
    return Vector3D(v.x * math.cos(a) - v.y * math.sin(a),
                    v.x * math.sin(a) + v.y * math.cos(a), v.z)


def flank_state(simulation: Any, faction: str, spec: FlankSpec) -> FlankState:
    """The faction's group assignment and this checkpoint's stations
    (computed once per sim time, shared by its captains)."""
    now = float(getattr(simulation, "current_time", 0.0))
    with _LOCK:
        store = getattr(simulation, _ATTR, None)
        if store is None:
            store = {}
            setattr(simulation, _ATTR, store)
        st = store.get(faction)
        if st is None:
            st = FlankState()
            store[faction] = st
        if st.time == now:
            return st
        ships = [s for s in simulation.ships.values() if _alive(s)]
        friends = [s for s in ships if s.faction == faction]
        enemies = [s for s in ships if s.faction != faction]
        if not st.groups:
            st.groups = assign_groups(friends, spec)
        st.time = now
        st.stations = {}
        if not friends or not enemies:
            return st
        eguns = [e for e in enemies if tactics.has_guns(e)] or enemies
        ce = _centre(eguns)
        holders = [s for s in friends if st.groups.get(s.ship_id) == "hold"] or friends
        ch = _centre(holders)
        axis = ch - ce
        if axis.magnitude < 1.0:
            axis = _centre(friends) - ce
        axis = axis.normalized()
        st.axis = axis
        sign = {"flank_a": 1.0, "flank_b": -1.0}
        for g in ("flank_a", "flank_b"):
            st.stations[g] = ce + _rot_z(axis, sign[g] * spec.angle_deg) * (spec.station_km * 1000.0)
        if spec.kind == "split":
            boats_r = POSITION_RANGE_KM
            st.stations["boat_a"] = ce + _rot_z(axis, SPLIT_HALF_DEG) * (boats_r * 1000.0)
            st.stations["boat_b"] = ce + _rot_z(axis, -SPLIT_HALF_DEG) * (boats_r * 1000.0)
        # Attack trigger: every live flanker within tolerance of its station.
        if not st.attack_started and spec.kind in ("pincer", "envelop"):
            flankers = [s for s in friends if st.groups.get(s.ship_id, "").startswith("flank")]
            on = [on_station(s, st.stations[st.groups[s.ship_id]], ce, spec) for s in flankers]
            if ((flankers and sum(on) >= ATTACK_QUORUM * len(flankers))
                    or not flankers or now >= FLANK_TIMEOUT_S):
                st.attack_started = True
                st.attack_t = now
        return st


def on_station(ship: Any, station: Any, enemy_centre: Any, spec: FlankSpec) -> bool:
    """On the station's BEARING from the enemy (+-STATION_TOL_DEG) at about
    its range (STATION_RANGE_BAND_KM): the station moves with the enemy, so
    a point tolerance alone was met in round 1 only by the timeout."""
    if (ship.position - station).magnitude / 1000.0 <= STATION_TOL_KM:
        return True
    want = (station - enemy_centre).normalized()
    have = (ship.position - enemy_centre)
    rng = have.magnitude / 1000.0
    if rng <= 0:
        return False
    ang = math.degrees(math.acos(max(-1.0, min(1.0, want.dot(have.normalized())))))
    lo, hi = STATION_RANGE_BAND_KM
    return ang <= STATION_TOL_DEG and spec.station_km - lo <= rng <= spec.station_km + hi


def plan_text(spec: Optional[FlankSpec], st: Optional["FlankState"]) -> str:
    """The battle plan as the flag ship announces it on the FLEET NET."""
    if spec is None:
        return ("Battle plan: close and pass. Heavy line ahead - close nose-first, "
                "coast through his line, re-close; focus fire.")
    if spec.kind == "wall":
        return (f"Battle plan: wall. Close slowly to {WALL_ENGAGE_KM:.0f} km, noses on the "
                f"nearest gun, engines off, fire when optimal.")
    n = sum(1 for g in st.groups.values() if g.startswith("flank"))
    if spec.kind == "pincer":
        return (f"Battle plan: pincer. {n} fast hulls to {spec.angle_deg:.0f} deg off his facing "
                f"at {spec.station_km:.0f} km; the line holds at {HOLD_WAIT_KM:.0f} km until they "
                f"are on station.")
    return (f"Battle plan: double envelopment. {n} ships in two groups to +-{spec.angle_deg:.0f} deg "
            f"at {spec.station_km:.0f} km.")


def point_km(v: Any) -> Dict[str, float]:
    return {"x": round(v.x / 1000.0, 2), "y": round(v.y / 1000.0, 2), "z": round(v.z / 1000.0, 2)}


def wall_maneuver(range_km: float, closing_kps: float, engage_km: float = WALL_ENGAGE_KM) -> str:
    """'INTERCEPT' (close, under the closure cap) or 'PADLOCK' (coast, nose on)."""
    if range_km > engage_km and closing_kps < WALL_CLOSE_KPS:
        return "INTERCEPT"
    return "PADLOCK"
