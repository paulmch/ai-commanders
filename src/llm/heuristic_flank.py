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


FLANK_DOCTRINES: Dict[str, FlankSpec] = {
    "wall": FlankSpec("wall"),
    "pincer45": FlankSpec("pincer", 45.0),
    "pincer70": FlankSpec("pincer", 70.0),
    "pincer90": FlankSpec("pincer", 90.0),
    "pincer70c": FlankSpec("pincer", 70.0, CUT_CORNER_RANGE_KM),
    "envelop70": FlankSpec("envelop", 70.0),
    "split": FlankSpec("split"),
}


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


def point_km(v: Any) -> Dict[str, float]:
    return {"x": round(v.x / 1000.0, 2), "y": round(v.y / 1000.0, 2), "z": round(v.z / 1000.0, 2)}


def wall_maneuver(range_km: float, closing_kps: float, engage_km: float = WALL_ENGAGE_KM) -> str:
    """'INTERCEPT' (close, under the closure cap) or 'PADLOCK' (coast, nose on)."""
    if range_km > engage_km and closing_kps < WALL_CLOSE_KPS:
        return "INTERCEPT"
    return "PADLOCK"
