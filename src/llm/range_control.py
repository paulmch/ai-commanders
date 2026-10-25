"""
Range control for commanders: closest approach, stopping distance and
precomputed WORLD-frame headings ready to pass to set_heading.

Newtonian combat has no brakes. Two fleets closing at 4.6 km/s need
v^2/(2a) of room to stop - 359 km for a 3 g corvette, 1078 km for a 1 g
battleship - so by the time the enemy "looks close" it is far too late, and
every maneuver that is not a deliberate burn against the relative velocity
ends in a merge: a pass at a kilometre or two, a few seconds of fire, and
two fleets flying apart. Commanders (LLM or human) should not have to do the
vector algebra to avoid that. This module does it for them, per enemy
group:

* closing speed, time and distance of the closest approach if they coast,
  and their own stopping time/distance at full deceleration (plus the flip),
* a plain merge warning when a pass inside MERGE_KM is coming,
* ready-made headings with a one-line effect estimate each - cancel closure
  (hold range), open range, re-engage (after a pass, when the range is
  opening), pass wide (turn a merge into a wide pass), flank left/right - and
  the paired attitude-only option (nose back on the enemy to shoot while
  coasting; for a gunless torpedo group, the launch window),
* formation throttles: one heading for several hulls only holds formation
  if every hull accelerates like the slowest one.

Estimates are constant-acceleration kinematics of the relative motion: the
friendly side turns (bang-bang, the engine's own turn-time estimator), then
burns along the heading; each enemy group keeps its current thrust until the
next checkpoint and coasts after that. HEADING in the engine turns first and
burns once nose-on (see simulation.HEADING_BURN_CONE_DEG), which is what makes
these numbers come true (tests/test_range_control_and_evade.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from ..physics import G_STANDARD, Vector3D
    from ..geometry import NOSE_HIT_ANGLE_THRESHOLD, TAIL_HIT_ANGLE_THRESHOLD
except ImportError:  # legacy imports with src/ on sys.path
    from physics import G_STANDARD, Vector3D
    from geometry import NOSE_HIT_ANGLE_THRESHOLD, TAIL_HIT_ANGLE_THRESHOLD


# Enemies within this distance of each other (single linkage) are one group.
GROUP_LINK_KM = 150.0
# A closest approach inside this is a merge: the fleets fly through each other.
MERGE_KM = 50.0
# How far ahead closest approaches are searched for.
CPA_HORIZON_S = 900.0
# Integration step of the relative-motion estimate (the closest approach is
# refined inside each step, so this only bounds the acceleration error).
PREDICT_STEP_S = 2.0
# "Pass wide" burns just long enough to make the pass this wide (or as wide
# as it still can before the pass), and never longer than PASS_WIDE_MAX_BURN_S.
PASS_WIDE_TARGET_KM = 150.0
PASS_WIDE_MAX_BURN_S = 300.0
# The merge warning fires when braking has to start within this many
# decision intervals (or is already too late).
MERGE_WARN_INTERVALS = 2.0
# Estimates assume each enemy keeps its current thrust this long, then
# coasts: long enough that a charging line is not mistaken for a coasting
# one while we brake (the dangerous error), short enough that a 15-minute
# closest-approach search does not extrapolate one burn forever.
ENEMY_ACCEL_HOLD_S = 120.0
# Enemy groups that get the full option block (the most urgent one always;
# the next only if it threatens a merge or is inside DETAIL_RANGE_KM); the
# rest get their status lines only.
FULL_DETAIL_GROUPS = 2
DETAIL_RANGE_KM = 1000.0
# Relative speeds below this count as matched (nothing to cancel).
MATCHED_MPS = 10.0
# "Re-engage" (offered once the range is OPENING, i.e. after a pass): burn
# until the relative velocity is a gentle closure of this speed along the
# line of sight - slow enough to stop again, fast enough to matter. Torpedo
# launches need closure >= 0 to keep a solution; guns want 1-3 km/s.
REENGAGE_CLOSING_KPS = 1.0
# Weapon arcs (angle between the nose and the target line).
SPINAL_ARC_DEG = 30.0
TURRET_ARC_DEG = 90.0

# Names an admiral / MCP client may use for "every ship in my fleet".
FLEET_ALIASES = frozenset({"all", "all ships", "all_ships", "fleet", "whole fleet",
                           "entire fleet", "*", "everyone"})

_Z = Vector3D(0.0, 0.0, 1.0)
_Y = Vector3D(0.0, 1.0, 0.0)


def is_fleet_alias(name: Any) -> bool:
    """True when a ship name addresses the whole fleet ("all ships", ...)."""
    return isinstance(name, str) and name.strip().lower() in FLEET_ALIASES


# ---------------------------------------------------------------------------
# Ship helpers
# ---------------------------------------------------------------------------

def ship_accel_ms2(ship: Any) -> float:
    """Full-throttle acceleration the ship can produce now (engine damage in)."""
    ks = ship.kinematic_state
    eff = ship.get_effective_thrust_fraction() if hasattr(ship, "get_effective_thrust_fraction") else 1.0
    return ks.max_acceleration_ms2() * eff


def _current_accel(ship: Any) -> Vector3D:
    """Thrust acceleration the ship is applying right now (its torch is visible)."""
    ks = ship.kinematic_state
    throttle = getattr(ship, "applied_throttle", 0.0) or 0.0
    if ks.mass_kg <= 0 or throttle <= 0:
        return Vector3D(0.0, 0.0, 0.0)
    return ks.forward * (ks.thrust_n * throttle / ks.mass_kg)


def _turn_time(sim: Any, ships: Sequence[Any], direction: Vector3D, engines_on: bool) -> float:
    """Longest bang-bang turn (from rest) any of ``ships`` needs to face ``direction``."""
    worst = 0.0
    for ship in ships:
        angle = ship.forward.angle_to(direction)
        worst = max(worst, sim._estimate_turn_time(ship, angle, engines_on=engines_on))
    return worst


def _burn_start(sim: Any, ships: Sequence[Any], direction: Vector3D) -> float:
    """
    Seconds until a HEADING burn along ``direction`` lights up for the whole
    group: each ship turns (thrust-vectored bang-bang) and starts burning as
    soon as its nose is inside HEADING_BURN_CONE_DEG, i.e. the last
    sqrt(2*cone/alpha) seconds of its turn already burn. The slowest ship sets
    the time (formations hold everyone's burn until then).
    """
    from ..simulation import HEADING_BURN_CONE_DEG
    cone = math.radians(HEADING_BURN_CONE_DEG)
    worst = 0.0
    for ship in ships:
        angle = ship.forward.angle_to(direction)
        if angle <= cone:
            continue
        turn = sim._estimate_turn_time(ship, angle, engines_on=True)
        alpha, _ = sim._get_rotation_params(ship, True)
        early = math.sqrt(2.0 * cone / alpha) if alpha > 0 else 0.0
        worst = max(worst, max(0.0, turn - early))
    return worst


def formation_throttles(ships: Sequence[Any], throttle: float) -> Dict[str, float]:
    """
    Per-hull throttle so every ship accelerates like the slowest one at
    ``throttle``: t_i = throttle * a_slowest / a_i. Same heading + these
    throttles = same acceleration vector = the formation holds.
    """
    accels = {s.ship_id: ship_accel_ms2(s) for s in ships}
    live = [a for a in accels.values() if a > 0]
    if not live:
        return {sid: 0.0 for sid in accels}
    a_min = min(live)
    return {sid: (min(1.0, max(0.0, throttle) * a_min / a) if a > 0 else 0.0)
            for sid, a in accels.items()}


def formation_heading_maneuvers(
    sim: Any,
    ships: Sequence[Any],
    heading: Any,
    throttle: float,
    duration: Optional[float] = None,
) -> Dict[str, Any]:
    """
    One HEADING order for several hulls that keeps their formation.

    Each hull gets the throttle matched to the slowest hull's acceleration,
    and the burn starts for everyone once the slowest turner is nose-on
    (Maneuver.burn_start_time): a corvette that finished its flip first
    would otherwise burn for tens of seconds alone and leave the formation.
    ``heading`` is a {x,y,z} dict or a Vector3D (world frame). Throttle 0 is
    attitude-only for everyone (no burn, no sync needed).
    """
    from ..simulation import Maneuver, ManeuverType

    h = _as_vector(heading)
    if h is None:
        raise ValueError("heading must be a non-zero {x, y, z} vector")
    h = h.normalized()
    live = [s for s in ships
            if not s.is_destroyed and not getattr(s, "is_dying", False)
            and not getattr(s, "is_surrendered", False)]
    if duration is None:
        duration = getattr(sim, "decision_interval", 30.0)
    throttles = formation_throttles(live, throttle)
    burn_start = None
    if throttle > 0 and live:
        burn_start = sim.current_time + _burn_start(sim, live, h)
    heading_dict = {"x": round(h.x, 4), "y": round(h.y, 4), "z": round(h.z, 4)}
    return {
        s.ship_id: Maneuver(
            maneuver_type=ManeuverType.HEADING,
            start_time=sim.current_time,
            duration=duration,
            throttle=round(throttles.get(s.ship_id, 0.0), 4),
            heading_direction=dict(heading_dict),
            burn_start_time=burn_start,
        )
        for s in live
    }


def _as_vector(value: Any) -> Optional[Vector3D]:
    if value is None:
        return None
    if isinstance(value, Vector3D):
        v = value
    elif isinstance(value, dict):
        try:
            v = Vector3D(float(value.get("x", 0.0)), float(value.get("y", 0.0)),
                         float(value.get("z", 0.0)))
        except (TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        v = Vector3D(*(float(c) for c in value))
    else:
        return None
    return v if v.magnitude > 1e-9 else None


# ---------------------------------------------------------------------------
# Relative-motion estimate
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    """Relative motion (friendly minus enemy) under piecewise-constant thrust."""
    min_range_m: float
    min_t_s: float
    probes: Dict[float, Tuple[Vector3D, Vector3D]] = field(default_factory=dict)


def _propagate(
    p: Vector3D,
    v: Vector3D,
    own_segments: Sequence[Tuple[float, float, Vector3D]],
    enemy_accel: Vector3D,
    enemy_until_s: float,
    horizon_s: float = CPA_HORIZON_S,
    probe_times: Iterable[float] = (),
) -> _Track:
    """
    Fly the relative state p (own - enemy position, m) and v (m/s) forward.

    own_segments: (t0, t1, accel) windows of friendly thrust; the enemy keeps
    ``enemy_accel`` until ``enemy_until_s`` and coasts after. Returns the
    closest approach inside the horizon (refined within each step) and the
    relative state at every probe time.
    """
    probes = sorted({round(t, 6) for t in probe_times if 0.0 <= t <= horizon_s})
    cuts = {0.0, horizon_s, min(enemy_until_s, horizon_s)}
    for t0, t1, _ in own_segments:
        cuts.update((min(max(t0, 0.0), horizon_s), min(max(t1, 0.0), horizon_s)))
    cuts.update(probes)
    edges = sorted(cuts)

    px, py, pz = p.x, p.y, p.z
    vx, vy, vz = v.x, v.y, v.z
    best = math.sqrt(px * px + py * py + pz * pz)
    best_t = 0.0
    out: Dict[float, Tuple[Vector3D, Vector3D]] = {}
    if 0.0 in probes:
        out[0.0] = (Vector3D(px, py, pz), Vector3D(vx, vy, vz))

    for a_t, b_t in zip(edges[:-1], edges[1:]):
        if b_t - a_t <= 1e-9:
            continue
        mid = 0.5 * (a_t + b_t)
        ax = ay = az = 0.0
        for t0, t1, acc in own_segments:
            if t0 <= mid < t1:
                ax += acc.x
                ay += acc.y
                az += acc.z
        if mid < enemy_until_s:
            ax -= enemy_accel.x
            ay -= enemy_accel.y
            az -= enemy_accel.z
        coasting = ax == 0.0 and ay == 0.0 and az == 0.0
        t = a_t
        while t < b_t - 1e-9:
            # Unaccelerated motion is a straight line: one exact step.
            h = (b_t - t) if coasting else min(PREDICT_STEP_S, b_t - t)
            # Closest approach inside this step (ballistic refinement).
            vv = vx * vx + vy * vy + vz * vz
            if vv > 1e-12:
                tau = -(px * vx + py * vy + pz * vz) / vv
                if 0.0 < tau < h:
                    qx = px + vx * tau + 0.5 * ax * tau * tau
                    qy = py + vy * tau + 0.5 * ay * tau * tau
                    qz = pz + vz * tau + 0.5 * az * tau * tau
                    d = math.sqrt(qx * qx + qy * qy + qz * qz)
                    if d < best:
                        best, best_t = d, t + tau
            px += vx * h + 0.5 * ax * h * h
            py += vy * h + 0.5 * ay * h * h
            pz += vz * h + 0.5 * az * h * h
            vx += ax * h
            vy += ay * h
            vz += az * h
            t += h
            d = math.sqrt(px * px + py * py + pz * pz)
            if d < best:
                best, best_t = d, t
        key = round(b_t, 6)
        if key in probes:
            out[key] = (Vector3D(px, py, pz), Vector3D(vx, vy, vz))
    return _Track(best, best_t, out)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class HeadingOption:
    """One precomputed heading and what it does."""
    key: str            # cancel | open | reengage | wide | left | right | face
    label: str
    heading: Vector3D   # unit, world frame
    throttle: float     # 0 = attitude only
    turn_s: float       # nose onto the heading first
    burn_s: float       # then burn this long at `throttle`
    effect: str
    aspect: str = ""    # which way the nose points during the burn
    cpa_km: Optional[float] = None
    cpa_t_s: Optional[float] = None
    end_range_km: Optional[float] = None
    rel_speed_end_kps: Optional[float] = None

    def heading_str(self) -> str:
        h = self.heading
        return f"({h.x:+.2f},{h.y:+.2f},{h.z:+.2f})"

    def to_dict(self) -> Dict[str, Any]:
        h = self.heading
        d: Dict[str, Any] = {
            "option": self.key,
            "heading": {"x": round(h.x, 3), "y": round(h.y, 3), "z": round(h.z, 3)},
            "throttle": round(self.throttle, 2),
            "turn_s": round(self.turn_s, 1),
            "burn_s": round(self.burn_s, 1),
            "effect": self.effect,
            "nose": self.aspect,
        }
        for k in ("cpa_km", "cpa_t_s", "end_range_km", "rel_speed_end_kps"):
            val = getattr(self, k)
            if val is not None:
                d[k] = round(val, 2 if k == "rel_speed_end_kps" else 1)
        return d


@dataclass
class GroupRangeInfo:
    """Range-control picture against one enemy group."""
    label: str
    names: List[str]
    ship_ids: List[str]
    range_km: float
    closing_kps: float          # + closing, - opening
    rel_speed_kps: float
    cpa_km: float               # if the friendly side coasts
    cpa_t_s: float
    stop_s: float               # relative stop at full decel, flip excluded
    stop_km: float
    flip_s: float
    accel_g: float
    warning: Optional[str] = None
    options: List[HeadingOption] = field(default_factory=list)
    cpa_ship: str = ""          # which enemy hull passes closest (groups)

    def option(self, key: str) -> Optional[HeadingOption]:
        return next((o for o in self.options if o.key == key), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.label,
            "ships": list(self.ship_ids),
            "names": list(self.names),
            "range_km": round(self.range_km, 1),
            "closing_kps": round(self.closing_kps, 2),
            "closest_approach_km": round(self.cpa_km, 1),
            "closest_approach_in_s": round(self.cpa_t_s, 0),
            "closest_ship": self.cpa_ship or (self.names[0] if self.names else ""),
            "stop_s": round(self.stop_s, 0),
            "stop_km": round(self.stop_km, 0),
            "flip_s": round(self.flip_s, 0),
            "warning": self.warning,
            "options": [o.to_dict() for o in self.options],
        }


def stopping(closing_mps: float, accel_ms2: float) -> Tuple[float, float]:
    """(seconds, metres) to kill a closing speed at full deceleration (no flip)."""
    if closing_mps <= 0 or accel_ms2 <= 0:
        return 0.0, 0.0
    return closing_mps / accel_ms2, closing_mps * closing_mps / (2.0 * accel_ms2)


def group_enemies(own_pos: Vector3D, enemies: Sequence[Any],
                  link_km: float = GROUP_LINK_KM) -> List[List[Any]]:
    """Single-linkage groups of enemies, nearest group first."""
    link_m = link_km * 1000.0
    groups: List[List[Any]] = []
    for enemy in sorted(enemies, key=lambda e: (e.position - own_pos).magnitude):
        touching = [g for g in groups
                    if any((enemy.position - m.position).magnitude <= link_m for m in g)]
        if not touching:
            groups.append([enemy])
            continue
        merged = touching[0]
        merged.append(enemy)
        for g in touching[1:]:
            merged.extend(g)
            groups.remove(g)
    return groups


def _mean(vectors: Sequence[Vector3D]) -> Vector3D:
    n = max(1, len(vectors))
    return Vector3D(sum(v.x for v in vectors) / n, sum(v.y for v in vectors) / n,
                    sum(v.z for v in vectors) / n)


def _flank_axes(los: Vector3D) -> Tuple[Vector3D, Vector3D]:
    """(left, right) perpendicular to the line of sight, world +z up."""
    right = los.cross(_Z)
    if right.magnitude < 0.1:
        right = los.cross(_Y)
    right = right.normalized()
    return right * -1.0, right


def _nose_aspect(heading: Vector3D, los: Vector3D, ships: Sequence[Any], detail: bool,
                 duration_s: Optional[float] = None) -> str:
    """
    Which face the enemy sees while the nose points along ``heading`` (the
    drive thrusts along the nose, so a burn dictates the attitude), for how
    long, and whether the guns can bear meanwhile.
    """
    angle = math.degrees(heading.angle_to(los))
    if angle < NOSE_HIT_ANGLE_THRESHOLD:
        face, section = "NOSE to enemy", "nose"
    elif angle > TAIL_HIT_ANGLE_THRESHOLD:
        face, section = "TAIL to enemy", "tail"
    else:
        face, section = "BROADSIDE to enemy", "lateral"
    if duration_s is not None and duration_s > 0:
        face += f" ~{duration_s:.0f} s"
    if detail and len(ships) == 1:
        armor = getattr(ships[0], "armor", None)
        sec = armor.get_section(section) if armor else None
        if sec is not None:
            face += f" ({section} armor {sec.thickness_cm:.0f} cm exposed)"
    guns = _gun_arcs(ships)
    if guns is None:
        return face
    has_spinal, has_turret = guns
    bears = []
    if has_spinal and angle <= SPINAL_ARC_DEG:
        bears.append("spinal")
    if has_turret and angle <= TURRET_ARC_DEG:
        bears.append("turrets")
    if bears:
        return f"{face}, {' + '.join(bears)} bear"
    return f"{face}, guns masked"


def _gun_arcs(ships: Sequence[Any]) -> Optional[Tuple[bool, bool]]:
    """(any spinal, any turret) among the ships' guns; None if none carry guns."""
    spinal = turret = False
    for ship in ships:
        for slot, ws in (getattr(ship, "weapons", None) or {}).items():
            if str(slot).startswith("pd"):
                continue
            weapon = getattr(ws, "weapon", None)
            if weapon is None or getattr(weapon, "is_missile", False):
                continue
            if getattr(weapon, "is_turreted", True):
                turret = True
            else:
                spinal = True
    if not (spinal or turret):
        return None
    return spinal, turret


def analyze_range_control(
    sim: Any,
    own_ships: Sequence[Any],
    enemies: Optional[Sequence[Any]] = None,
    interval_s: Optional[float] = None,
    detail: bool = True,
) -> List[GroupRangeInfo]:
    """
    Range-control picture of ``own_ships`` (one ship, or a fleet moving as
    one: centre of mass, slowest hull's acceleration, slowest turn) against
    every enemy group. ``detail`` adds per-hull armor numbers (single ship).
    """
    own = [s for s in own_ships
           if not s.is_destroyed and not getattr(s, "is_dying", False)]
    if not own:
        return []
    if enemies is None:
        enemies = sim.get_enemy_ships(own[0].ship_id)
    enemies = [e for e in enemies if not e.is_destroyed and not getattr(e, "is_dying", False)]
    if not enemies:
        return []
    if interval_s is None:
        interval_s = float(getattr(sim, "decision_interval", 30.0) or 30.0)

    own_pos = _mean([s.position for s in own])
    own_vel = _mean([s.velocity for s in own])
    accel = min(ship_accel_ms2(s) for s in own)

    infos: List[GroupRangeInfo] = []
    for members in group_enemies(own_pos, enemies):
        info = _analyze_group(sim, own, own_pos, own_vel, accel, members,
                              interval_s, detail)
        infos.append(info)

    # Most urgent first: an impending merge, then the nearest.
    infos.sort(key=lambda g: (g.warning is None, g.cpa_t_s if g.warning else g.range_km))
    for i, info in enumerate(infos):
        info.label = f"GROUP {chr(ord('A') + i)}" if i < 26 else f"GROUP {i + 1}"
    return infos


def _analyze_group(sim, own, own_pos, own_vel, accel, members, interval_s, detail):
    e_pos = _mean([m.position for m in members])
    e_vel = _mean([m.velocity for m in members])
    e_acc = _mean([_current_accel(m) for m in members])
    p = own_pos - e_pos                   # enemy centre -> us
    v = own_vel - e_vel                   # our velocity relative to them
    rng = p.magnitude
    los = (p * -1.0).normalized() if rng > 1.0 else Vector3D(1.0, 0.0, 0.0)  # us -> them
    closing = v.dot(los)                  # + closing

    # Closest approaches are judged ship-to-ship (the group centre can pass
    # wide while its nearest hull flies straight through us); the range a
    # burn settles at is judged on the centre.
    pairs = [(o.position - m.position, o.velocity - m.velocity, _current_accel(m),
              getattr(m, "name", None) or m.ship_id)
             for o in own for m in members]

    def evaluate(segs, probe: Optional[float] = None, enemy_until: float = ENEMY_ACCEL_HOLD_S,
                 horizon: float = CPA_HORIZON_S):
        best, best_t, who = float("inf"), 0.0, ""
        for pp, vv, aa, name in pairs:
            tr = _propagate(pp, vv, segs, aa, enemy_until, horizon_s=horizon)
            if tr.min_range_m < best:
                best, best_t, who = tr.min_range_m, tr.min_t_s, name
        centre = _propagate(p, v, segs, e_acc, enemy_until,
                            probe_times=[probe] if probe is not None else [])
        state = centre.probes.get(round(probe, 6), (p, v)) if probe is not None else (p, v)
        return best, best_t, who, state

    base_m, base_t, base_who, _ = evaluate([])
    info = GroupRangeInfo(
        label="",
        names=[getattr(m, "name", None) or m.ship_id for m in members],
        ship_ids=[m.ship_id for m in members],
        range_km=rng / 1000.0,
        closing_kps=closing / 1000.0,
        rel_speed_kps=v.magnitude / 1000.0,
        cpa_km=base_m / 1000.0,
        cpa_t_s=base_t,
        stop_s=0.0, stop_km=0.0, flip_s=0.0,
        accel_g=accel / G_STANDARD,
    )
    info.cpa_ship = base_who if len(members) > 1 else ""
    if accel <= 0:
        return info

    # ---- cancel closure: burn against the relative velocity -------------
    # Solve for the heading h and burn time T that bring the relative
    # velocity to zero while the enemy keeps its current thrust:
    #   v1 + (h*a*throttle - a_e)*T = 0,  v1 = velocity when the burn starts.
    # Against a charging enemy the heading leans into his acceleration and the
    # burn is longer than |v|/a; holding afterwards means matching his burn.
    e_acc_mag = e_acc.magnitude
    h, t_f, solved = None, 0.0, None
    dv = v.magnitude
    # After a pass the same burn stops the range OPENING: name it for what it
    # does then ("hold range"), not "cancel closure" of a closure that is gone.
    cancel_label = "cancel closure" if closing >= 0 else "hold range"
    attempted = dv > MATCHED_MPS or e_acc_mag > 0.05 * accel
    if attempted:
        h = (v * -1.0).normalized() if dv > 1e-6 else (e_acc.normalized() if e_acc_mag > 0 else los * -1.0)
        for _ in range(2):                 # turn time depends on the heading
            t_f = _burn_start(sim, own, h)
            v1 = v - e_acc * t_f
            solved = _cancel_solve(v1, e_acc, accel, interval_s - t_f)
            if solved is None:
                break
            h, throttle, burn = solved
    if solved is not None:
        end = t_f + burn
        # Consistent with the solve: he keeps burning while we brake.
        # Closest approach only up to the relative stop: afterwards the
        # range holds for as long as the order is renewed.
        m_min, m_t, _, (pe, ve) = evaluate([(t_f, end, h * (accel * throttle))], probe=end,
                                           enemy_until=max(ENEMY_ACCEL_HOLD_S, end),
                                           horizon=end)
        hold_km = pe.magnitude / 1000.0
        when = _span(end, interval_s)
        effect = f"rel. speed ~0 after {end:.0f} s, range then holds ~{hold_km:.0f} km"
        if e_acc_mag > 0.02 * G_STANDARD:
            effect += f" while you keep matching his {e_acc_mag / G_STANDARD:.1f}g burn"
        if m_min / 1000.0 < 0.8 * hold_km and m_t > 1.0:
            effect += f" (passes {m_min / 1000.0:.0f} km first)"
        cancel_opt = HeadingOption(
            key="cancel", label=cancel_label, heading=h, throttle=throttle,
            turn_s=t_f, burn_s=burn, effect=f"{when}: {effect}",
            aspect=_nose_aspect(h, los, own, detail, end),
            cpa_km=m_min / 1000.0, cpa_t_s=m_t,
            end_range_km=hold_km, rel_speed_end_kps=ve.magnitude / 1000.0)
        info.options.append(cancel_opt)
        info.flip_s = t_f
    elif attempted:
        # The solve failed: no burn of ours can null the relative velocity
        # because he accelerates harder along the line than we can. This used
        # to print "already matched ... the range holds" right under the merge
        # warning that said the opposite (audit ADM-09). No heading is given.
        info.options.append(HeadingOption(
            key="cancel", label=cancel_label, heading=los * -1.0, throttle=0.0,
            turn_s=0.0, burn_s=0.0,
            effect=(f"IMPOSSIBLE: he out-accelerates you along the line "
                    f"({e_acc_mag / G_STANDARD:.1f}g burn vs your {accel / G_STANDARD:.1f}g; "
                    f"rel. speed {dv:.0f} m/s) - only pass wide / flank changes the merge"),
            rel_speed_end_kps=dv / 1000.0))
    else:
        info.options.append(HeadingOption(
            key="cancel", label=cancel_label, heading=los * -1.0, throttle=0.0,
            turn_s=0.0, burn_s=0.0,
            effect=f"already matched (rel. speed {dv:.0f} m/s): coast and the range holds",
            end_range_km=rng / 1000.0, rel_speed_end_kps=dv / 1000.0))

    stop_s, stop_m = stopping(max(0.0, closing), accel)
    info.stop_s, info.stop_km = stop_s, stop_m / 1000.0

    # ---- open range: one checkpoint's burn straight away from them -------
    away = los * -1.0
    t_f = _burn_start(sim, own, away)
    burn = interval_s
    end = t_f + burn
    m_min, m_t, _, (pe, ve) = evaluate([(t_f, end, away * accel)], probe=end)
    los_e = (pe * -1.0).normalized() if pe.magnitude > 1.0 else los
    c_e = ve.dot(los_e) / 1000.0
    rate = f"closing {c_e:.2f}" if c_e > 0 else f"opening {-c_e:.2f}"
    info.options.append(HeadingOption(
        key="open", label="open range", heading=away, throttle=1.0, turn_s=t_f,
        burn_s=burn,
        effect=f"{_span(end, interval_s)}: then {pe.magnitude / 1000.0:.0f} km, {rate} km/s",
        aspect=_nose_aspect(away, los, own, detail, end),
        cpa_km=m_min / 1000.0, cpa_t_s=m_t, end_range_km=pe.magnitude / 1000.0))

    # ---- re-engage: after a pass, turn the opening into a gentle closure --
    if closing < 0:
        reengage = _reengage_option(sim, own, p, v, e_acc, los, accel, interval_s, detail,
                                    evaluate)
        if reengage is not None:
            info.options.append(reengage)

    # ---- pass wide: push the existing miss wider --------------------------
    left, right = _flank_axes(los)
    if closing > 0 and base_t > 0:
        lat = v - los * v.dot(los)
        wide = lat.normalized() if lat.magnitude > 1.0 else left
        t_f = _burn_start(sim, own, wide)
        max_burn = max(0.0, min(PASS_WIDE_MAX_BURN_S, max(interval_s, base_t - t_f)))

        def wide_cpa(burn: float):
            return evaluate([(t_f, t_f + burn, wide * accel)] if burn > 0 else [])

        # Shortest burn that makes the pass PASS_WIDE_TARGET_KM wide (or the
        # widest pass still possible when that is out of reach).
        burn = max_burn
        m_min, m_t, _, _ = wide_cpa(burn)
        if m_min / 1000.0 > PASS_WIDE_TARGET_KM:
            lo, hi = 0.0, max_burn
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                if wide_cpa(mid)[0] / 1000.0 >= PASS_WIDE_TARGET_KM:
                    hi = mid
                else:
                    lo = mid
            burn = max(1.0, math.ceil(hi))
            m_min, m_t, _, _ = wide_cpa(burn)
        info.options.append(HeadingOption(
            key="wide", label="pass wide", heading=wide, throttle=1.0, turn_s=t_f,
            burn_s=burn,
            effect=(f"{_span(t_f + burn, interval_s)}: closest approach ~{m_min / 1000.0:.0f} km "
                    f"in {m_t:.0f} s (coasting: {info.cpa_km:.1f} km)"),
            aspect=_nose_aspect(wide, los, own, detail, t_f + burn),
            cpa_km=m_min / 1000.0, cpa_t_s=m_t))

    # ---- flank left / right: one checkpoint's burn across his line -------
    for key, axis in (("left", left), ("right", right)):
        t_f = _burn_start(sim, own, axis)
        burn = interval_s
        end = t_f + burn
        m_min, m_t, _, (pe, _) = evaluate([(t_f, end, axis * accel)], probe=end)
        coast_pe, _ = _centre_probe(p, v, e_acc, ENEMY_ACCEL_HOLD_S, end)
        side_km = (pe - coast_pe).magnitude / 1000.0
        info.options.append(HeadingOption(
            key=key, label=f"flank {key}", heading=axis, throttle=1.0, turn_s=t_f,
            burn_s=burn,
            effect=(f"{_span(end, interval_s)}: {side_km:.0f} km off your coasting line, "
                    f"closest approach {m_min / 1000.0:.0f} km"),
            aspect=_nose_aspect(axis, los, own, detail, end),
            cpa_km=m_min / 1000.0, cpa_t_s=m_t))

    # ---- coast & shoot: attitude only, nose back on them -----------------
    # A gunless (torpedo) group has nothing to "shoot" while coasting; its
    # version of the same attitude-only option is the launch window - and the
    # one condition that decides it is that the range is not opening.
    tv = _turn_time(sim, own, los, engines_on=True)
    rcs = _turn_time(sim, own, los, engines_on=False)
    turn_text = ("nose already on him" if rcs < 1.0 else
                 f"RCS turn ~{rcs:.0f} s, no drift; PADLOCK ~{tv:.0f} s (token burn, few m/s)")
    if _gun_arcs(own) is None:
        window = ("closing: torpedoes launched now keep their solution" if closing >= 0 else
                  "range OPENING: launches now chase a receding target - re-engage first")
        info.options.append(HeadingOption(
            key="face", label="coast & launch", heading=los, throttle=0.0,
            turn_s=rcs, burn_s=0.0, effect=f"{turn_text}; {window}",
            aspect=_nose_aspect(los, los, own, detail).replace(" exposed)", ")")))
    else:
        info.options.append(HeadingOption(
            key="face", label="coast & shoot", heading=los, throttle=0.0,
            turn_s=rcs, burn_s=0.0, effect=turn_text,
            aspect=_nose_aspect(los, los, own, detail).replace(" exposed)", ")")))

    # ---- merge warning ----------------------------------------------------
    if closing > 0 and info.cpa_km < MERGE_KM and base_t < CPA_HORIZON_S:
        # Room to kill the closure: flip, then decelerate at what is left of
        # our acceleration after his burn toward us.
        his_push = max(0.0, e_acc.dot(p.normalized())) if rng > 1.0 else 0.0
        decel = accel - his_push
        need_km = (closing * info.flip_s) / 1000.0 + (
            closing * closing / (2.0 * decel) / 1000.0 if decel > 0 else float("inf"))
        slack_s = (info.range_km - need_km) / (closing / 1000.0)
        at = f"{info.cpa_km:.1f} km" + (f" ({info.cpa_ship})" if info.cpa_ship else "")
        if decel <= 0:
            info.warning = (
                f"MERGE IN {base_t:.0f} s at {at}: he out-accelerates you toward you - "
                f"braking cannot stop it; pass wide or flank NOW.")
        elif slack_s <= 0:
            info.warning = (
                f"MERGE IN {base_t:.0f} s at {at}, too late to brake "
                f"(need {need_km:.0f} km incl. flip, have {info.range_km:.0f} km); "
                f"pass wide or burn laterally NOW.")
        elif slack_s <= MERGE_WARN_INTERVALS * interval_s:
            info.warning = (
                f"MERGE IN {base_t:.0f} s at {at} unless you act: braking must start "
                f"within {slack_s:.0f} s (need {need_km:.0f} km incl. flip, have "
                f"{info.range_km:.0f} km). Cancel closure now, or pass wide.")
    return info


def _reengage_option(sim, own, p, v, e_acc, los, accel, interval_s, detail, evaluate):
    """
    'Re-engage' heading for an OPENING range (after a pass): burn so the
    relative velocity becomes a closure of REENGAGE_CLOSING_KPS along the line
    of sight, lateral drift nulled. Reports when the range stops opening, how
    far out that is, and the closure once the burn ends.
    """
    target_v = los * (REENGAGE_CLOSING_KPS * 1000.0)   # our velocity rel. to them, wanted
    need = target_v - v
    if need.magnitude < MATCHED_MPS or accel <= 0:
        return None
    h = need.normalized()
    t_f = _burn_start(sim, own, h)
    burn = need.magnitude / accel
    end = t_f + burn
    segs = [(t_f, end, h * accel)]
    # Sample the burn to find where the opening turns into closure.
    step = max(1.0, end / 60.0)
    probes = [k * step for k in range(int(end / step) + 2)]
    track = _propagate(p, v, segs, e_acc, ENEMY_ACCEL_HOLD_S, horizon_s=max(end, 1.0),
                       probe_times=probes)
    peak_km, peak_t = p.magnitude / 1000.0, 0.0
    for t, (pp, _) in sorted(track.probes.items()):
        if pp.magnitude / 1000.0 > peak_km:
            peak_km, peak_t = pp.magnitude / 1000.0, t
    _, _, _, (pe, ve) = evaluate(segs, probe=end)
    los_e = (pe * -1.0).normalized() if pe.magnitude > 1.0 else los
    c_e = ve.dot(los_e) / 1000.0
    effect = (f"{_span(end, interval_s)}: range stops opening after ~{peak_t:.0f} s at "
              f"~{peak_km:.0f} km, then closing {c_e:.1f} km/s from {pe.magnitude / 1000.0:.0f} km")
    return HeadingOption(
        key="reengage", label="re-engage", heading=h, throttle=1.0, turn_s=t_f,
        burn_s=burn, effect=effect, aspect=_nose_aspect(h, los, own, detail, end),
        end_range_km=pe.magnitude / 1000.0, rel_speed_end_kps=ve.magnitude / 1000.0)


def _cancel_solve(v1: Vector3D, e_acc: Vector3D, accel: float,
                  window_s: float) -> Optional[Tuple[Vector3D, float, float]]:
    """
    Heading, throttle and burn time that null the relative velocity ``v1``
    against an enemy accelerating at ``e_acc``: h*a*thr = e_acc - v1/T.

    Full throttle: |e_acc - v1*u| = a with u = 1/T, a quadratic in u. If that
    burn is shorter than ``window_s`` (the rest of this checkpoint) the burn
    is stretched over the window at reduced throttle instead, so a renewed
    order does not overshoot. None when the enemy out-accelerates us along
    the needed direction (no solution).
    """
    vv = v1.dot(v1)
    if vv < 1e-9:
        if e_acc.magnitude <= 0 or accel <= 0:
            return None
        thr = min(1.0, e_acc.magnitude / accel)
        return e_acc.normalized(), thr, max(window_s, 1.0)
    b = e_acc.dot(v1)
    c = e_acc.dot(e_acc) - accel * accel
    disc = b * b - vv * c
    if disc < 0:
        return None
    u = (b + math.sqrt(disc)) / vv
    if u <= 0:
        return None
    burn = 1.0 / u
    thr = 1.0
    if window_s > 1.0 and burn < window_s - 1.0:
        burn = window_s
        u = 1.0 / burn
    need = e_acc - v1 * u
    mag = need.magnitude
    if mag <= 0:
        return None
    thr = min(1.0, mag / accel)
    return need * (1.0 / mag), thr, burn


def _centre_probe(p: Vector3D, v: Vector3D, e_acc: Vector3D, enemy_until_s: float,
                  t: float) -> Tuple[Vector3D, Vector3D]:
    """Relative state (centre) at ``t`` if the friendly side coasts."""
    track = _propagate(p, v, [], e_acc, enemy_until_s, horizon_s=max(t, 1e-6),
                       probe_times=[t])
    return track.probes.get(round(t, 6), (p, v))


def _span(seconds: float, interval_s: float) -> str:
    """'~85 s = renew 3 checkpoints' / 'this checkpoint'."""
    n = max(1, math.ceil(seconds / interval_s - 1e-6))
    if n == 1:
        return "this checkpoint"
    return f"~{seconds:.0f} s ({n} checkpoints, renew it)"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _option_line(opt: HeadingOption, target_name: str, verb: str = "set_heading") -> str:
    if opt.key == "face":
        return (f"  {opt.label:<14} {verb}{opt.heading_str()} throttle 0 or PADLOCK "
                f"{target_name}: {opt.effect} | {opt.aspect}")
    if opt.throttle <= 0:
        return f"  {opt.label:<14} {opt.effect}"
    return (f"  {opt.label:<14} {verb}{opt.heading_str()} @{opt.throttle * 100:.0f}% "
            f"{opt.effect} | {opt.aspect}")


def _group_header(info: GroupRangeInfo) -> str:
    names = ", ".join(info.names[:4]) + (" ..." if len(info.names) > 4 else "")
    rate = (f"closing {info.closing_kps:.2f} km/s" if info.closing_kps >= 0
            else f"opening {-info.closing_kps:.2f} km/s")
    return (f"{info.label}: {names} ({len(info.names)} ship{'s' if len(info.names) != 1 else ''})"
            f" - {info.range_km:.0f} km, {rate}")


_SUBJECT = {"Your": "you coast", "Fleet": "the fleet coasts", "Ship": "it coasts"}
_STOPPER = {"Your": "Your", "Fleet": "Fleet", "Ship": "Its"}


def _group_status(info: GroupRangeInfo, who: str) -> str:
    ship = f" ({info.cpa_ship})" if info.cpa_ship else ""
    cpa = (f"closest approach {info.cpa_km:.1f} km{ship} in {info.cpa_t_s:.0f} s"
           if info.cpa_t_s > 0.5 else "range not closing (closest is now)")
    stop = ""
    if info.closing_kps > 0 and info.stop_s > 0:
        stop = (f" {_STOPPER.get(who, who)} stop: {info.stop_s:.0f} s / {info.stop_km:.0f} km at "
                f"{info.accel_g:.1f}g (+{info.flip_s:.0f} s flip).")
    subject = _SUBJECT.get(who, "the fleet coasts")
    return f"  If {subject}: {cpa}.{stop}"


def _detailed(i: int, info: GroupRangeInfo, max_detail: int) -> bool:
    """The most urgent group always gets options; the next only if it matters."""
    return i < max_detail and (i == 0 or bool(info.warning) or info.range_km <= DETAIL_RANGE_KM)


def format_range_control(
    infos: Sequence[GroupRangeInfo],
    who: str = "Your",
    max_detail: int = FULL_DETAIL_GROUPS,
    where: Optional[str] = None,
) -> str:
    """
    Compact prompt block. ``who`` is "Your" (captain), "Fleet" (admiral,
    fleet centre) or "Ship" (the admiral's order for one ship).
    """
    if not infos:
        return ""
    verb = "set_heading" if who == "Your" else "heading"
    if where is None:
        where = {"Your": "set_heading",
                 "Ship": "issue_order heading= for this ship"}.get(
                     who, "issue_order heading= (fleet centre; see FORMATION)")
    lines = [f"=== RANGE CONTROL (world-frame headings for {where}; the drive "
             "thrusts along the NOSE) ==="]
    for i, info in enumerate(infos):
        lines.append(_group_header(info))
        lines.append(_group_status(info, who))
        if info.warning:
            lines.append(f"  !! {info.warning}")
        if not _detailed(i, info, max_detail):
            continue
        target = info.names[0] if info.names else "target"
        for opt in info.options:
            lines.append(_option_line(opt, target, verb))
    return "\n".join(lines)


def format_heading_options(
    infos: Sequence[GroupRangeInfo],
    max_detail: int = FULL_DETAIL_GROUPS,
    verb: str = "heading",
) -> str:
    """
    Only the precomputed heading options per group (the admiral's turn state
    states each group's range, closest approach and stop data in its own
    GEOMETRY block, so they are not repeated here).
    """
    lines = []
    for i, info in enumerate(infos):
        if not _detailed(i, info, max_detail):
            continue
        target = info.names[0] if info.names else "target"
        lines.append(f"{info.label} ({target}{' +' + str(len(info.names) - 1) if len(info.names) > 1 else ''}):")
        for opt in info.options:
            lines.append(_option_line(opt, target, verb))
    return "\n".join(lines)


def formation_line(live: Sequence[Any], mcp: bool = False) -> str:
    """The FORMATION advice: per-hull throttles that hold the formation."""
    if len(live) <= 1:
        return ""
    thr = formation_throttles(live, 1.0)
    slowest = min(live, key=ship_accel_ms2)
    parts = ", ".join(f"{getattr(s, 'name', None) or s.ship_id} {thr[s.ship_id] * 100:.0f}%"
                      for s in live)
    return (
        f"FORMATION: headings above are for the fleet centre (slowest hull "
        f"{ship_accel_ms2(slowest) / G_STANDARD:.2f}g sets the pace, slowest turn sets the "
        f"flip). Same heading with equal acceleration keeps the formation: throttles "
        f"for 100% of the slowest hull = {parts}. " + (
            "Easiest: ONE set_maneuver with ship_id \"all\", HEADING, heading_direction "
            "and throttle" if mcp else
            "Easiest: ONE issue_order with ship_name=\"all ships\", heading and throttle")
        + " - each hull's throttle is scaled and the burn starts together.")


def captain_range_control(sim: Any, ship: Any) -> str:
    """Range-control block for one ship's captain ("" when nothing to show)."""
    try:
        infos = analyze_range_control(sim, [ship], detail=True)
    except Exception:  # advisory only: never let it kill a decision
        return ""
    return format_range_control(infos, who="Your")


def admiral_range_control(sim: Any, ships: Sequence[Any], mcp: bool = False) -> str:
    """Range-control block for an admiral: fleet centre vs enemy groups."""
    live = [s for s in ships if not s.is_destroyed and not getattr(s, "is_dying", False)]
    if not live:
        return ""
    try:
        infos = analyze_range_control(sim, live, detail=False)
    except Exception:  # advisory only
        return ""
    if not infos:
        return ""
    where = ("set_maneuver HEADING heading_direction (fleet centre; ship_id \"all\" = "
             "formation)" if mcp else None)
    lines = [format_range_control(infos, who="Fleet", where=where)]
    formation = formation_line(live, mcp)
    if formation:
        lines.append(formation)
    return "\n".join(lines)


def mcp_range_control(sim: Any, ships: Sequence[Any]) -> Dict[str, Any]:
    """Structured range control for MCP clients (plus the same text)."""
    live = [s for s in ships if not s.is_destroyed and not getattr(s, "is_dying", False)]
    if not live:
        return {}
    infos = analyze_range_control(sim, live, detail=False)
    return {
        "frame": "world; pass heading as heading_direction to set_maneuver HEADING",
        "fleet_groups": [g.to_dict() for g in infos],
        "formation_throttles": {k: round(v, 3) for k, v in formation_throttles(live, 1.0).items()},
        "text": admiral_range_control(sim, live, mcp=True),
    }
