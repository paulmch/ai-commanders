"""
Torpedo launch sequence (pass 5): a round leaves its rack or cell, drifts
clear on its manoeuvring thrusters, turns onto its first guidance heading and
only then lights its torch.

Game design, not engineering: the numbers below are tuned for how the replay
looks and for consistent timing (scripts/launch_sequence_facts.py measures
them per hull class and launch geometry).

1. EJECT (LAUNCH_EJECT_S). The round starts at its own rack round / VLS cell
   (data/hull_mounts.json ``rounds``: body frame x forward, y port, z dorsal,
   generated from the 3D hull kit; which one: pick_round, the round on the
   target's side of the hull first), with the ship's velocity, the velocity of
   that point of a spinning hull (omega x r) and LAUNCH_EJECT_SPEED_MPS along
   the mount's eject direction (rack: outboard; cell: out of the hatch). Its
   nose is the round's nose at rest, turned with the ship's attitude (roll
   included), and it keeps the ship's spin. No thruster fires.
2. SEPARATE. The lateral RCS (the round's own, rcs_thrust_n: 5 % of the main
   drive, ~0.6 g on the Trident) pushes it off the line through its ship
   along its heading until it is clear (below), and it turns its nose onto
   the first guidance heading at up to LAUNCH_TURN_RATE_DEG_S
   (LAUNCH_TURN_ACCEL_DEG_S2 to start and stop a turn; it brakes the spin it
   inherited first). The seeker tracks its target from the rail: the heading
   is what the round's guidance would command now
   (TorpedoGuidance._collision_course_guidance; the line of sight for other
   guidance modes), so the target-acceleration estimator sees only the
   TARGET's velocity - never the ejection kick.
3. BOOST. The torch lights at the start of the first tick on which, measured
   on the start-of-tick state of the ships (CombatSimulation takes a snapshot
   before the ships move): the round is LAUNCH_CLEAR_M from its own hull
   (capsule of ShipGeometry length and radius) and LAUNCH_FRIENDLY_CLEAR_M
   from every other friendly hull within LAUNCH_FRIENDLY_SCAN_M, its exhaust
   cone (LAUNCH_EXHAUST_CONE_DEG half angle, out to LAUNCH_EXHAUST_RANGE_M)
   misses its own hull, and its nose is within LAUNCH_ALIGN_DEG of the
   heading. After LAUNCH_TIMEOUT_S it lights as soon as it is LAUNCH_CLEAR_M
   from its own hull, whatever else holds; at LAUNCH_HARD_TIMEOUT_S it lights
   regardless. A round whose ship is destroyed during the sequence has no
   hull to clear (it still needs the other conditions). On its first boost
   tick the torch first cancels the sideways drift the sequence gave it
   (CombatSimulation._release_torpedo).

Every rule is a pure function of the round and the start-of-tick snapshot: no
random draws, no ship read after it moved, nothing that depends on insertion
order (the friendly check is an all() over a dict). Recording-only state
(phase, nose) rides on the round for the replay.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from .physics import Vector3D
    from . import weapon_arcs
except ImportError:  # pragma: no cover - script-style imports
    from physics import Vector3D
    import weapon_arcs


# =============================================================================
# Constants (game design)
# =============================================================================

# Ejection: a spring / gas push off the rail or out of the cell. The corvette's
# rail rounds sit 7.8-10.4 m off its axis (its drive bell: 6.1 m); a 3 g
# corvette slides its stern past a round in ~1.7 s, by when 10 m/s has carried
# the round 17 m further out, and the RCS push keeps it off the plume axis
# (measured: 0 s inside the 7.5 deg plume cone; scripts/launch_sequence_facts.py).
# With the RCS it reaches LAUNCH_CLEAR_M in 3-4 s, about the time a quarter
# turn takes: a faster ejection would not light the torch sooner.
LAUNCH_EJECT_SPEED_MPS = 10.0
# Eject phase: no thruster fires, the round only drifts.
LAUNCH_EJECT_S = 1.0
# Clear distance from the own hull's surface before the torch may light.
LAUNCH_CLEAR_M = 50.0
# ... and from every other friendly hull (cheap: only hulls this near count).
LAUNCH_FRIENDLY_CLEAR_M = 50.0
LAUNCH_FRIENDLY_SCAN_M = 2_000.0
# The round's own exhaust cone must not contain its own hull (half angle,
# plus the hull's angular radius from the round), out to this range.
LAUNCH_EXHAUST_CONE_DEG = 15.0
LAUNCH_EXHAUST_RANGE_M = 5_000.0
# Nose within this of the first guidance heading.
LAUNCH_ALIGN_DEG = 10.0
# Attitude: the Trident's turn on its RCS (Terra Invicta torpedoes turn at
# 20-30 deg/s, src/torpedo.py from_terra_invicta).
LAUNCH_TURN_RATE_DEG_S = 30.0
LAUNCH_TURN_ACCEL_DEG_S2 = 30.0
# After this the round lights as soon as it is LAUNCH_CLEAR_M from its hull...
LAUNCH_TIMEOUT_S = 8.0
# ... and at this whatever holds.
LAUNCH_HARD_TIMEOUT_S = 15.0
# Measurement switch (AIC_LAUNCH_SEQUENCE=0: the old instant ignition at the
# ship's centre); CombatSimulation.launch_sequence is the per-battle flag.
LAUNCH_SEQUENCE_DEFAULT = os.environ.get("AIC_LAUNCH_SEQUENCE", "1") != "0"

PHASE_EJECT = "eject"
PHASE_SEPARATE = "separate"
PHASE_BOOST = "boost"

_RATE = math.radians(LAUNCH_TURN_RATE_DEG_S)
_ALPHA = math.radians(LAUNCH_TURN_ACCEL_DEG_S2)
_ALIGN = math.radians(LAUNCH_ALIGN_DEG)
_CONE = math.radians(LAUNCH_EXHAUST_CONE_DEG)
_EPS_T = 1e-9


# =============================================================================
# State
# =============================================================================

@dataclass
class LaunchSequence:
    """A round's launch sequence (on Torpedo.launch_seq). Once released it
    stays attached in PHASE_BOOST (the replay reads ignition_t)."""
    t0: float                       # launch time (s)
    source_id: str
    faction: str
    mount: Optional[str]            # launcher slot (hull_mounts), None: no data
    round_index: int                # round of that mount (firing order)
    nose: Vector3D                  # world unit nose
    omega: Vector3D                 # world angular velocity (rad/s)
    eject_dir: Vector3D             # world unit eject direction
    phase: str = PHASE_EJECT
    hold_forecast_s: float = 0.0    # forecast launch-to-ignition (s), at launch
    heading: Optional[Vector3D] = None   # first guidance heading, last tick
    ignition_t: Optional[float] = None
    reason: Optional[str] = None    # clear | free | timeout | hard_timeout

    @property
    def active(self) -> bool:
        return self.phase != PHASE_BOOST


class ShipSnap:
    """Start-of-tick picture of one hull for the launch rules."""
    __slots__ = ("ship_id", "faction", "pos", "vel", "fwd", "half_len", "radius", "gone")

    def __init__(self, ship: Any) -> None:
        self.ship_id = ship.ship_id
        self.faction = ship.faction
        p, f, v = ship.position, ship.forward, ship.velocity
        self.pos = Vector3D(p.x, p.y, p.z)
        self.vel = Vector3D(v.x, v.y, v.z)
        self.fwd = Vector3D(f.x, f.y, f.z)
        geom = getattr(ship, "geometry", None)
        length = float(getattr(geom, "length_m", 100.0) or 100.0)
        self.half_len = 0.5 * length
        self.radius = float(getattr(geom, "radius_m", length / 8.0) or length / 8.0)
        self.gone = bool(ship.is_destroyed)


def snapshot(sim: Any) -> Dict[str, ShipSnap]:
    """Every hull at the start of the tick (taken before the ships move)."""
    return {sid: ShipSnap(s) for sid, s in sim.ships.items()}


# =============================================================================
# Geometry helpers (pure)
# =============================================================================

def _closest_axis_point(snap: ShipSnap, p: Vector3D) -> Vector3D:
    rel = p - snap.pos
    s = max(-snap.half_len, min(snap.half_len, rel.dot(snap.fwd)))
    return snap.pos + snap.fwd * s


def hull_gap_m(snap: ShipSnap, p: Vector3D) -> float:
    """Distance from p to the hull capsule's surface (negative inside)."""
    return (p - _closest_axis_point(snap, p)).magnitude - snap.radius


def exhaust_on_hull(snap: ShipSnap, p: Vector3D, heading: Vector3D) -> bool:
    """True when the exhaust cone of a round at p burning along ``heading``
    contains part of the hull (sampled along its axis, each sample widened by
    its angular radius)."""
    back = heading * -1.0
    n = max(2, int(math.ceil(2.0 * snap.half_len / max(1.0, snap.radius))) + 1)
    for i in range(n):
        s = -snap.half_len + 2.0 * snap.half_len * i / (n - 1)
        v = snap.pos + snap.fwd * s - p
        d = v.magnitude
        if d <= snap.radius:
            return True
        if d > LAUNCH_EXHAUST_RANGE_M:
            continue
        c = max(-1.0, min(1.0, v.dot(back) / d))
        if math.acos(c) < _CONE + math.asin(min(1.0, snap.radius / d)):
            return True
    return False


def angle_between(a: Vector3D, b: Vector3D) -> float:
    na, nb = a.magnitude, b.magnitude
    if na <= 0 or nb <= 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, a.dot(b) / (na * nb))))


def _any_perpendicular(v: Vector3D) -> Vector3D:
    a = Vector3D(1.0, 0.0, 0.0) if abs(v.x) < 0.9 else Vector3D(0.0, 1.0, 0.0)
    return v.cross(a).normalized()


def rotate(v: Vector3D, omega: Vector3D, dt: float) -> Vector3D:
    """v turned by the angular velocity omega for dt (Rodrigues)."""
    w = omega.magnitude
    if w <= 0.0 or dt <= 0.0:
        return v
    k = omega * (1.0 / w)
    th = w * dt
    c, s = math.cos(th), math.sin(th)
    out = v * c + k.cross(v) * s + k * (k.dot(v) * (1.0 - c))
    return out.normalized()


def turn_time_s(angle_rad: float) -> float:
    """Time to turn the nose through angle_rad from rest and stop
    (bang-bang at LAUNCH_TURN_ACCEL_DEG_S2, capped at LAUNCH_TURN_RATE_DEG_S)."""
    return weapon_arcs.roll_time_s(angle_rad, _ALPHA, _RATE)


def rcs_accel(torp: Any) -> float:
    specs = torp.specs
    mass = float(getattr(torp, "current_mass_kg", specs.mass_kg) or specs.mass_kg)
    return float(specs.rcs_thrust_n) / max(1.0, mass)


def forecast_hold_s(turn_rad: float, rcs_mps2: float, dt: float = 1.0,
                    gap0_m: float = 0.0) -> float:
    """Forecast seconds from launch to ignition: the per-tick sequence flown
    from a COASTING ship (an accelerating one only clears sooner), the same
    discrete steps as advance()/check_release(): the round leaves the hull
    surface (``gap0_m`` off it) at LAUNCH_EJECT_SPEED_MPS, pushes outward on
    its RCS after the eject phase, and turns ``turn_rad`` from rest with the
    controller of _turn(); the torch lights at the first tick start with the
    clear distance made and the nose within LAUNCH_ALIGN_DEG (or the clear
    distance made at LAUNCH_TIMEOUT_S, or LAUNCH_HARD_TIMEOUT_S)."""
    dt = dt if dt > 0 else 1.0
    t, gap, v = 0.0, gap0_m, LAUNCH_EJECT_SPEED_MPS
    theta, w = max(0.0, turn_rad), 0.0
    while True:
        if t + _EPS_T >= LAUNCH_EJECT_S:
            if t + _EPS_T >= LAUNCH_HARD_TIMEOUT_S:
                return t
            if gap >= LAUNCH_CLEAR_M and (theta <= _ALIGN or t + _EPS_T >= LAUNCH_TIMEOUT_S):
                return t
        if t + _EPS_T >= LAUNCH_EJECT_S:
            rate = min(_RATE, math.sqrt(2.0 * _ALPHA * theta), theta / dt)
            w += max(-_ALPHA * dt, min(_ALPHA * dt, rate - w))
            theta = max(0.0, theta - w * dt)
            if gap < LAUNCH_CLEAR_M:
                v += max(0.0, rcs_mps2) * dt
        gap += v * dt
        t += dt


# =============================================================================
# Launch
# =============================================================================

def launcher_mount(sim: Any, ship: Any, launcher_index: int):
    """The hull_mounts launcher mount of the ship's ``launcher_index``-th
    tube (launchers in fleet_ships weapons order), or None."""
    table = getattr(sim, "hull_mounts", None) or {}
    hull = table.get(getattr(ship, "ship_type", None))
    if hull is None:
        return None
    tubes = [m for m in hull.mounts if m.kind == "launcher"]
    return tubes[launcher_index] if 0 <= launcher_index < len(tubes) else None


def pick_round(sim: Any, ship: Any, launcher_index: int, tube: Any, target: Any,
               time_s: float = 0.0) -> int:
    """Which of the tube's magazine rounds leaves next: of the rounds not yet
    fired, the one nearest the target's side of the hull (its body position
    along the line of sight), then the one farthest from the nearest
    friendly hull, then the lowest index. Geometry, not a fixed order, so a
    fleet and its mirror image (rotated OR reflected: fixed-fleet battles
    place the sides by reflection) fire mirror-image rounds."""
    mount = launcher_mount(sim, ship, launcher_index)
    if mount is None or not mount.rounds:
        return len(getattr(tube, "fired_rounds", None) or ())
    fired = set(getattr(tube, "fired_rounds", None) or ())
    left = [i for i in range(len(mount.rounds)) if i not in fired]
    if not left:
        return len(fired) % len(mount.rounds)
    frame = body_frame(ship, time_s)

    def body(v):
        n = v.magnitude
        return frame.to_body(v.x / n, v.y / n, v.z / n) if n > 0 else (0.0, 0.0, 0.0)
    los = body(target.position - ship.position) if target is not None else (0.0, 0.0, 0.0)
    mate = None
    best_d = math.inf
    for sid in sorted(getattr(sim, "ships", {}) or {}):
        s = sim.ships[sid]
        if s is ship or s.faction != ship.faction or s.is_destroyed:
            continue
        d = (s.position - ship.position).magnitude
        if d < best_d:
            best_d, mate = d, s
    away = body(ship.position - mate.position) if mate is not None else (0.0, 0.0, 0.0)

    def key(i):
        p = mount.rounds[i][0]
        toward = p[0] * los[0] + p[1] * los[1] + p[2] * los[2]
        off = p[0] * away[0] + p[1] * away[1] + p[2] * away[2]
        # metres, rounded to the millimetre: float noise between a fleet and
        # its mirror image must not reorder exact ties
        return (-round(toward, 3), -round(off, 3), i)
    return min(left, key=key)


def round_geometry(sim: Any, ship: Any, launcher_index: int, round_index: int
                   ) -> Tuple[Optional[str], Tuple[float, float, float],
                              Tuple[float, float, float], Tuple[float, float, float]]:
    """(slot, body pos m, body eject dir, body nose) of one magazine round.
    Without mount data: the dorsal surface amidships, ejected dorsal, nose
    forward."""
    mount = launcher_mount(sim, ship, launcher_index)
    if mount is not None and mount.rounds:
        pos, eject, nose = mount.rounds[round_index % len(mount.rounds)]
        return mount.slot, pos, eject, nose
    geom = getattr(ship, "geometry", None)
    r = float(getattr(geom, "radius_m", 10.0) or 10.0)
    if mount is not None:
        ax = mount.axis
        return mount.slot, mount.pos, ax, ax if abs(ax[0]) < 0.5 else (1.0, 0.0, 0.0)
    return None, (0.0, 0.0, r), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)


def body_frame(ship: Any, time_s: float = 0.0):
    """The ship's body frame now (navigation frame + hull roll)."""
    frame = weapon_arcs.ShipArcs(None, ship)
    arcs = getattr(ship, "arcs", None)
    frame.roll = float(getattr(arcs, "roll", 0.0) or 0.0)
    frame.set_frame(ship.forward, ship.kinematic_state.up, time_s)
    return frame


def ship_spin(ship: Any) -> Vector3D:
    w = float(getattr(ship, "angular_velocity_rad_s", 0.0) or 0.0)
    axis = getattr(ship, "angular_axis", None)
    if w <= 0.0 or axis is None or axis.magnitude <= 0.0:
        return Vector3D.zero()
    return axis.normalized() * w


def next_launch_nose(sim: Any, ship: Any, target: Any = None) -> Optional[Vector3D]:
    """World nose of the round the ship's next launch at ``target`` would
    eject (first ready tube), for forecasts; None without a ready tube."""
    tubes = list(getattr(ship, "ready_torpedo_launchers", None) or [])
    now = float(getattr(sim, "current_time", 0.0))
    for i, tube in enumerate(tubes):
        if tube.can_launch(now) if hasattr(tube, "can_launch") else True:
            k = pick_round(sim, ship, i, tube, target, now)
            _, _, _, nose = round_geometry(sim, ship, i, k)
            frame = body_frame(ship, now)
            return Vector3D(*frame.to_world(*nose))
    return None


def begin(sim: Any, ship: Any, launcher_index: int, torpedo: Any,
          target: Any = None) -> Dict[str, Any]:
    """Place a just-launched round on its rack / in its cell and start its
    sequence. Returns the torpedo_launched event fields (recording only)."""
    tubes = list(getattr(ship, "ready_torpedo_launchers", None) or [])
    tube = tubes[launcher_index] if 0 <= launcher_index < len(tubes) else None
    now = float(getattr(sim, "current_time", 0.0))
    k = pick_round(sim, ship, launcher_index, tube, target, now) if tube is not None else 0
    if tube is not None and hasattr(tube, "fired_rounds"):
        tube.fired_rounds.append(k)
    slot, pos_b, eject_b, nose_b = round_geometry(sim, ship, launcher_index, k)
    frame = body_frame(ship, now)
    r_w = Vector3D(*frame.to_world(*pos_b))
    e_w = Vector3D(*frame.to_world(*eject_b)).normalized()
    n_w = Vector3D(*frame.to_world(*nose_b)).normalized()
    omega = ship_spin(ship)
    torpedo.position = ship.position + r_w
    torpedo.velocity = ship.velocity + omega.cross(r_w) + e_w * LAUNCH_EJECT_SPEED_MPS
    torpedo.launch_position = Vector3D(torpedo.position.x, torpedo.position.y,
                                       torpedo.position.z)
    seq = LaunchSequence(
        t0=now, source_id=ship.ship_id, faction=ship.faction, mount=slot,
        round_index=k, nose=n_w, omega=omega, eject_dir=e_w)
    if target is not None:
        los = target.position - torpedo.position
        turn = angle_between(n_w, los) if los.magnitude > 0 else 0.0
        seq.hold_forecast_s = forecast_hold_s(
            turn, rcs_accel(torpedo), float(getattr(sim, "time_step", 1.0) or 1.0),
            hull_gap_m(ShipSnap(ship), torpedo.position))
    torpedo.launch_seq = seq
    return {
        "mount": slot,
        "round": k,
        "eject_pos_m": [round(c, 2) for c in pos_b],
        "eject_dir": [round(c, 4) for c in eject_b],
        "nose": [round(c, 4) for c in nose_b],
        "eject_speed_mps": LAUNCH_EJECT_SPEED_MPS,
    }


# =============================================================================
# Per tick
# =============================================================================

def check_release(seq: LaunchSequence, torp: Any, snaps: Dict[str, ShipSnap],
                  now: float) -> Optional[str]:
    """Why the torch lights at the start of this tick (None: not yet)."""
    el = now - seq.t0
    if el + _EPS_T < LAUNCH_EJECT_S:
        return None
    if el + _EPS_T >= LAUNCH_HARD_TIMEOUT_S:
        return "hard_timeout"
    p = torp.position
    own = snaps.get(seq.source_id)
    present = own is not None and not own.gone
    clear = (not present) or hull_gap_m(own, p) >= LAUNCH_CLEAR_M
    if not clear:
        return None
    if el + _EPS_T >= LAUNCH_TIMEOUT_S:
        return "timeout"
    h = seq.heading
    if h is None or angle_between(seq.nose, h) > _ALIGN:
        return None
    if present and exhaust_on_hull(own, p, h):
        return None
    for s in _friendlies_near(seq, p, snaps):
        if hull_gap_m(s, p) < LAUNCH_FRIENDLY_CLEAR_M:
            return None
    return "clear" if present else "free"


def _turn(seq: LaunchSequence, h: Vector3D, dt: float) -> None:
    n, w = seq.nose, seq.omega
    theta = angle_between(n, h)
    axis = n.cross(h)
    if axis.magnitude < 1e-9:
        axis = _any_perpendicular(n) if theta > 1.0 else Vector3D.zero()
    else:
        axis = axis.normalized()
    rate = min(_RATE, math.sqrt(2.0 * _ALPHA * theta), theta / max(dt, 1e-6))
    dw = axis * rate - w
    cap = _ALPHA * dt
    if dw.magnitude > cap:
        dw = dw.normalized() * cap
    w = w + dw
    seq.omega = w
    seq.nose = rotate(n, w, dt)


def rcs_push_direction(seq: LaunchSequence, hull: ShipSnap, p: Vector3D,
                       h: Vector3D) -> Vector3D:
    """Off the line through ``hull`` (the nearest one in the way) along the
    heading: the part of 'away from the hull' square to the heading (else the
    eject direction's, else any)."""
    away = p - _closest_axis_point(hull, p)
    perp = away - h * away.dot(h)
    if perp.magnitude > 0.05 * away.magnitude + 1e-3:
        return perp.normalized()
    e = seq.eject_dir - h * seq.eject_dir.dot(h)
    if e.magnitude > 1e-3:
        return e.normalized()
    return _any_perpendicular(h)


def _hull_in_the_way(seq: LaunchSequence, p: Vector3D, h: Vector3D,
                     snaps: Dict[str, ShipSnap]) -> Optional[ShipSnap]:
    """The hull the RCS pushes off: the nearest of the own hull (inside the
    clear distance or in the exhaust cone) and the friendly hulls inside
    their clear distance; None when nothing is in the way."""
    best, best_gap = None, math.inf
    own = snaps.get(seq.source_id)
    if own is not None and not own.gone:
        gap = hull_gap_m(own, p)
        if gap < LAUNCH_CLEAR_M or exhaust_on_hull(own, p, h):
            best, best_gap = own, gap
    for s in _friendlies_near(seq, p, snaps):
        gap = hull_gap_m(s, p)
        if gap < LAUNCH_FRIENDLY_CLEAR_M and gap < best_gap:
            best, best_gap = s, gap
    return best


def _friendlies_near(seq: LaunchSequence, p: Vector3D, snaps: Dict[str, ShipSnap]):
    scan2 = LAUNCH_FRIENDLY_SCAN_M * LAUNCH_FRIENDLY_SCAN_M
    for sid in sorted(snaps):
        s = snaps[sid]
        if s.gone or s.faction != seq.faction or s.ship_id == seq.source_id:
            continue
        d = s.pos - p
        if d.dot(d) <= scan2:
            yield s


def _closing_hull(seq: LaunchSequence, p: Vector3D, v: Vector3D, a_rcs: float,
                  snaps: Dict[str, ShipSnap], dt: float = 1.0) -> Optional[Vector3D]:
    """Brake direction when the round closes on a hull (its own, a turning
    or accelerating hull sweeping at it, or a friendly it was ejected toward)
    too fast to stop LAUNCH_CLEAR_M / 2 short of it on its RCS (counting the
    tick it flies before the next decision): straight away from the hull
    that leaves the least margin. None otherwise."""
    hulls = []
    own = snaps.get(seq.source_id)
    if own is not None and not own.gone:
        hulls.append(own)
    hulls.extend(_friendlies_near(seq, p, snaps))
    best, best_margin = None, math.inf
    for s in hulls:
        q = _closest_axis_point(s, p)
        off = p - q
        d = off.magnitude
        if d <= 1e-6:
            continue
        away = off * (1.0 / d)
        closing = -(v - s.vel).dot(away)
        if closing <= 0.0:
            continue
        margin = ((d - s.radius) - closing * closing / (2.0 * max(a_rcs, 1e-6))
                  - closing * dt)
        if margin < 0.5 * LAUNCH_CLEAR_M and margin < best_margin:
            best, best_margin = away, margin
    return best


def advance(seq: LaunchSequence, torp: Any, heading: Vector3D,
            snaps: Dict[str, ShipSnap], dt: float, now: float) -> float:
    """One tick of a round still in its sequence: attitude, RCS push, drift.
    Returns the RCS delta-v spent (km/s)."""
    seq.heading = heading
    el = now - seq.t0
    dv_kps = 0.0
    if el + _EPS_T < LAUNCH_EJECT_S:
        seq.phase = PHASE_EJECT
        seq.nose = rotate(seq.nose, seq.omega, dt)
    else:
        seq.phase = PHASE_SEPARATE
        _turn(seq, heading, dt)
        push = _closing_hull(seq, torp.position, torp.velocity, rcs_accel(torp), snaps, dt)
        if push is None:
            push_from = _hull_in_the_way(seq, torp.position, heading, snaps)
            if push_from is not None:
                push = rcs_push_direction(seq, push_from, torp.position, heading)
        if push is not None:
            before = torp.remaining_delta_v_kps
            torp.apply_lateral_thrust(push, dt, 1.0)
            dv_kps = before - torp.remaining_delta_v_kps
    torp.time_since_launch += dt
    torp.position = torp.position + torp.velocity * dt
    return dv_kps


def release(seq: LaunchSequence, now: float, reason: str) -> None:
    seq.phase = PHASE_BOOST
    seq.ignition_t = now
    seq.reason = reason
    seq.omega = Vector3D.zero()


# =============================================================================
# Forecasts
# =============================================================================

def hold_left_s(torp: Any) -> float:
    """Forecast seconds until this round's torch lights: its launch forecast
    less the time it has flown (at least one more tick while it is still in
    its sequence past that); 0 once it burns or without a sequence."""
    seq = getattr(torp, "launch_seq", None)
    if seq is None or not seq.active:
        return 0.0
    el = float(getattr(torp, "time_since_launch", 0.0))
    left = seq.hold_forecast_s - el
    return left if left > 1e-9 else 1.0


def predict_launch_hold_s(sim: Any, ship: Any, target: Any, specs: Any = None) -> float:
    """Forecast launch-to-ignition seconds for the ship's next round at
    ``target`` (its rest nose against the line of sight); 0 when the
    simulation flies no launch sequence."""
    if sim is not None and not getattr(sim, "launch_sequence", LAUNCH_SEQUENCE_DEFAULT):
        return 0.0
    nose = next_launch_nose(sim, ship, target) if sim is not None else None
    los = target.position - ship.position
    turn = angle_between(nose, los) if nose is not None and los.magnitude > 0 else 0.0
    if specs is None:
        tubes = list(getattr(ship, "ready_torpedo_launchers", None) or [])
        specs = tubes[0].specs if tubes else None
    a = (float(specs.rcs_thrust_n) / max(1.0, float(specs.mass_kg))) if specs is not None else 5.9
    dt = float(getattr(sim, "time_step", 1.0) or 1.0) if sim is not None else 1.0
    return forecast_hold_s(turn, a, dt)
