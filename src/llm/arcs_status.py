"""
Weapon-arc facts for commanders: what bears now, what is slewing, what would
unmask the rest (src/weapon_arcs.py). State, not tactics - the same lines go
to captains, admirals and MCP clients.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from .. import weapon_arcs
except ImportError:  # pragma: no cover - script-style imports
    import weapon_arcs


def _arcs(sim: Any, ship: Any):
    """The ship's ShipArcs with a fresh body frame, or None when arcs are off."""
    if not getattr(sim, "weapon_arcs", False) or not hasattr(sim, "_arcs_for"):
        return None
    try:
        arcs = sim._arcs_for(ship)
        ks = ship.kinematic_state
        arcs.set_frame(ks.forward, ks.up, getattr(sim, "current_time", 0.0))
    except Exception:  # noqa: BLE001 - advisory only (test doubles)
        return None
    return arcs if arcs.mounts else None


def _body_dir(arcs: Any, origin: Any, point: Any) -> Optional[Tuple[float, float, float]]:
    dx, dy, dz = point.x - origin.x, point.y - origin.y, point.z - origin.z
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n <= 0.0:
        return None
    return arcs.to_body(dx / n, dy / n, dz / n)


def pd_turrets_bearing(sim: Any, ship: Any, point: Any) -> Tuple[int, int]:
    """
    (bearing, operational): the ship's live PD turrets, and how many of them
    have a line of sight to ``point`` that their own hull leaves open. Without
    weapon arcs (or for a hull without mounts) every live turret bears.
    """
    turrets = [pd for pd in (getattr(ship, "point_defense", None) or [])
               if getattr(pd, "is_operational", True)]
    arcs = _arcs(sim, ship) if turrets else None
    body = _body_dir(arcs, ship.position, point) if arcs is not None else None
    if body is None:
        return len(turrets), len(turrets)
    cell = weapon_arcs.cell_index(*body)
    bearing = sum(1 for pd in turrets
                  if getattr(pd, "mount", None) is None or not pd.mount.mask[cell])
    return bearing, len(turrets)


_CONE_CACHE: Dict[Tuple[int, str], Dict[str, float]] = {}


def blind_cones(hull: Any) -> Dict[str, float]:
    """
    Half-angles (degrees) of the cones about the thrust axis inside which NO
    mount of a kind bears at ANY roll - roll turns a direction about that
    axis, so only a turn of the ship uncovers them. Keys 'pd_nose',
    'pd_stern', 'gun_nose', 'gun_stern'; 0 when the kind bears on the axis.
    Geometry of data/hull_mounts.json only, cached per hull class.
    """
    key = (id(hull), getattr(hull, "hull", ""))
    out = _CONE_CACHE.get(key)
    if out is not None:
        return out
    out = {}
    for kind in ("pd", "gun"):
        masks = [m.mask for m in hull.mounts if m.kind == kind]
        for end, sign in (("nose", 1.0), ("stern", -1.0)):
            cone = 0.0
            if masks:
                for step in range(0, 46):        # 0..45 degrees off the axis
                    a = math.radians(step)
                    bears = False
                    for k in range(360):         # every roll, 1 degree apart
                        p = math.radians(k)
                        cell = weapon_arcs.cell_index(sign * math.cos(a), math.sin(a) * math.cos(p),
                                                      math.sin(a) * math.sin(p))
                        if any(not mk[cell] for mk in masks):
                            bears = True
                            break
                    if bears:
                        break
                    cone = float(step)
            out[f"{kind}_{end}"] = cone
    _CONE_CACHE[key] = out
    return out


def _axis_angle(body: Tuple[float, float, float]) -> Tuple[str, float]:
    """('nose'|'stern', degrees off that end of the thrust axis)."""
    x = max(-1.0, min(1.0, body[0]))
    off_nose = math.degrees(math.acos(x))
    return ("nose", off_nose) if off_nose <= 90.0 else ("stern", 180.0 - off_nose)


def unmask_roll(sim: Any, ship: Any, mount: Any, body: Tuple[float, float, float]
                ) -> Optional[Tuple[float, float]]:
    """(roll in degrees, seconds) of the smallest roll that opens ``body`` for
    the mount, or None when no roll does (a direction near the roll axis)."""
    step = math.radians(weapon_arcs.ROLL_CANDIDATE_STEP_DEG)
    n = int(round(math.pi / step))
    bx, by, bz = body
    for k in range(1, n + 1):
        for phi in (k * step, -k * step):
            c, s = math.cos(phi), math.sin(phi)
            if not mount.mask[weapon_arcs.cell_index(bx, by * c + bz * s, bz * c - by * s)]:
                accel, rate = sim._arcs_roll_params(ship)
                return math.degrees(abs(phi)), weapon_arcs.roll_time_s(phi, accel, rate)
    return None


def gun_report(sim: Any, ship: Any, target: Any) -> Optional[Dict[str, Any]]:
    """
    Per gun mount of ``ship`` against ``target``: does it bear, how long until
    it is on (slew), and what roll unmasks it. None when arcs are off or the
    hull has no gun mounts.
    """
    arcs = _arcs(sim, ship)
    if arcs is None or not arcs.guns:
        return None
    los = _body_dir(arcs, ship.position, target.position)
    if los is None:
        return None
    rows: List[Dict[str, Any]] = []
    for slot, mount in arcs.guns.items():
        ws = ship.weapons.get(slot)
        if ws is None or not getattr(ws, "is_operational", True):
            continue
        tracking = mount.target_id == target.ship_id and mount.has_want
        want = (mount.wx, mount.wy, mount.wz) if tracking and mount.kind == "gun" else los
        row: Dict[str, Any] = {"slot": slot, "kind": mount.kind}
        if mount.kind == "spinal":
            off = math.degrees(math.acos(max(-1.0, min(1.0, los[0]))))
            row["bears"] = off <= ws.weapon.pivot_range_deg
            row["off_axis_deg"] = off
            row["slew_s"] = 0.0
        else:
            row["bears"] = not mount.mask[weapon_arcs.cell_index(*want)]
            if row["bears"]:
                dot = max(-1.0, min(1.0, mount.ax * want[0] + mount.ay * want[1] + mount.az * want[2]))
                row["slew_s"] = (0.0 if (tracking and mount.state == weapon_arcs.ON_TARGET)
                                 or not mount.trained else math.acos(dot) / mount.rate)
            else:
                row["unmask"] = unmask_roll(sim, ship, mount, want)
                if row["unmask"] is None and arcs.hull is not None:
                    end, off = _axis_angle(want)
                    if off <= blind_cones(arcs.hull)[f"gun_{end}"]:
                        row["cone"] = (end, off)
        rows.append(row)
    if not rows:
        return None
    return {"mounts": rows, "bearing": sum(1 for r in rows if r["bears"]), "total": len(rows)}


def mount_bears(sim: Any, ship: Any, slot: str, target: Any) -> bool:
    """True unless weapon arcs mask ``slot`` of ``ship`` from ``target`` now."""
    report = gun_report(sim, ship, target)
    if report is None:
        return True
    for row in report["mounts"]:
        if row["slot"] == slot:
            return bool(row["bears"]) if row["kind"] == "gun" else True
    return True


def gun_arcs_text(sim: Any, ship: Any, target: Any) -> Optional[str]:
    """'2 of 3 gun mounts bear; weapon_2 masked by own hull, clears after a
    45 deg roll (~60 s); weapon_3 on in 4 s'."""
    report = gun_report(sim, ship, target)
    if report is None:
        return None
    parts = [f"{report['bearing']} of {report['total']} gun mounts bear"]
    for row in report["mounts"]:
        if row["kind"] == "spinal":
            if not row["bears"]:
                parts.append(f"{row['slot']} spinal {row['off_axis_deg']:.0f} deg off the nose "
                             f"(outside its gimbal: turn the ship)")
        elif row["bears"]:
            if row["slew_s"] >= 1.0:
                parts.append(f"{row['slot']} slewing, on in {row['slew_s']:.0f} s")
        elif row.get("unmask"):
            deg, secs = row["unmask"]
            parts.append(f"{row['slot']} masked by own hull, clears after a "
                         f"{deg:.0f} deg roll (~{secs:.0f} s)")
        elif row.get("cone"):
            end, off = row["cone"]
            parts.append(f"{row['slot']} masked: the target is {off:.0f} deg off the {end}, inside "
                         f"the hull's blind cone (no battery bears there at any roll) - turn the ship")
        else:
            parts.append(f"{row['slot']} masked by own hull, no roll clears it "
                         f"(change heading)")
    return "; ".join(parts)


def arcs_status_lines(sim: Any, ship: Any, with_blind_cones: bool = True) -> List[str]:
    """
    Own-ship weapon-arc state for a status block: guns against the primary
    target, PD against the nearest hostile live torpedo. Empty when arcs are
    off or there is nothing to say. ``with_blind_cones=False`` leaves out the
    per-class blind-cone line (the captain has it in its cached doctrine,
    blind_cone_doctrine).
    """
    lines: List[str] = []
    arcs = _arcs(sim, ship)
    if arcs is None:
        return lines
    target_id = getattr(ship, "primary_target_id", None)
    target = sim.get_ship(target_id) if target_id else None
    if target is not None and not (target.is_destroyed or target.is_dying):
        text = gun_arcs_text(sim, ship, target)
        if text:
            lines.append(f"ARCS vs {getattr(target, 'name', target.ship_id)}: {text}")
    nearest = None
    for flight in getattr(sim, "torpedoes", None) or []:
        if getattr(flight, "is_disabled", False):
            continue
        source = sim.get_ship(flight.source_ship_id)
        if source is not None and source.faction == ship.faction:
            continue
        d = flight.torpedo.position.distance_to(ship.position)
        if nearest is None or d < nearest[0]:
            nearest = (d, flight)
    cones = blind_cones(arcs.hull) if arcs.hull is not None else {}
    if nearest is not None:
        bearing, live = pd_turrets_bearing(sim, ship, nearest[1].torpedo.position)
        if live:
            line = (f"ARCS PD: {bearing} of {live} turrets bear on the nearest hostile "
                    f"torpedo ({nearest[0] / 1000.0:.0f} km)")
            body = _body_dir(arcs, ship.position, nearest[1].torpedo.position)
            if bearing == 0 and body is not None:
                end, off = _axis_angle(body)
                if off <= cones.get(f"pd_{end}", 0.0):
                    line += (f": it is {off:.0f} deg off the {end}, inside the hull's PD blind "
                             f"cone - no roll helps, turn the ship")
            lines.append(line)
    blind = [f"{kind} within {cones[f'{kind}_{end}']:.0f} deg of the {end}"
             for kind in ("pd", "gun") for end in ("nose", "stern")
             if cones.get(f"{kind}_{end}", 0.0) >= 1.0]
    if blind and with_blind_cones:
        lines.append("ARCS blind cones (no mount of that kind bears at any roll; only a turn "
                     "uncovers them): " + ", ".join(b.replace("pd ", "PD ").replace("gun ", "guns ")
                                                    for b in blind))
    roll_deg = math.degrees(arcs.roll_goal - arcs.roll)
    if abs(roll_deg) >= 1.0:
        lines.append(f"ARCS roll: rolling {abs(roll_deg):.0f} deg more to unmask (automatic)")
    return lines


def arcs_doctrine_numbers(fleet_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    The numbers the doctrine quotes about weapon arcs, from the engine's
    constants and data/hull_mounts.json (bound by tests/test_doctrine_accuracy.py).
    """
    table = weapon_arcs.load_hull_mounts()
    fractions = []
    for hull in table.values():
        for mount in hull.mounts:
            if mount.kind in ("gun", "pd"):
                fractions.append(1.0 - sum(mount.mask) / float(len(mount.mask)))
    open_pct = 100.0 * sum(fractions) / len(fractions) if fractions else 100.0
    roll_90_s = 0.0
    rcs = (((fleet_data or {}).get("ships", {}).get("cruiser", {}).get("attitude_control") or {})
           .get("rcs") or {})
    if rcs:
        k = weapon_arcs.ROLL_RCS_AUTHORITY_FACTOR
        roll_90_s = weapon_arcs.roll_time_s(
            math.pi / 2.0, math.radians(rcs["angular_accel_deg_s2"]) * k,
            math.radians(rcs["max_angular_velocity_deg_s"]) * k)
    return {
        "open_sky_pct": open_pct,
        "battery_deg_s": weapon_arcs.BATTERY_TRAVERSE_DEG_S,
        "heavy_deg_s": weapon_arcs.HEAVY_BATTERY_TRAVERSE_DEG_S,
        "pd_deg_s": weapon_arcs.PD_TRAVERSE_DEG_S,
        "cruiser_roll_90_s": roll_90_s,
    }


def blind_cone_doctrine(ship_type: Optional[str]) -> str:
    """
    The static weapon-arc facts of one hull class for the captain's cached
    doctrine (the per-turn "ARCS blind cones" line moved here). Worded so it
    cannot read as "guns are stern-arc only" (the admiral audit found that
    misreading 7 times): the cones are where NO mount bears.
    """
    hull = weapon_arcs.load_hull_mounts().get(ship_type or "")
    if hull is None:
        return ""
    cones = blind_cones(hull)
    has_guns = any(m.kind == "gun" for m in hull.mounts)
    has_spinal = any(m.kind == "spinal" for m in hull.mounts)
    lines = ["WEAPON ARCS (your hull class): every mount fires only where your own hull leaves "
             "it clear, and the ship rolls itself to unmask them (automatic)."]
    for kind, label in (("gun", "gun battery"), ("pd", "PD turret")):
        if kind == "gun" and not has_guns:
            continue
        where = [f"within {cones[f'{kind}_{end}']:.0f} deg of {'dead ahead' if end == 'nose' else 'dead astern'}"
                 for end in ("nose", "stern") if cones.get(f"{kind}_{end}", 0.0) >= 1.0]
        if where:
            spinal = (" (your spinal fires dead ahead)" if kind == "gun" and has_spinal
                      and cones.get("gun_nose", 0.0) >= 1.0 else "")
            lines.append(f"- No {label} can fire at a target {' or '.join(where)}, at any roll - "
                         f"only turning the ship opens that cone{spinal}; in every other "
                         f"direction at least one {label} bears at some roll.")
        else:
            lines.append(f"- Your {label.replace('battery', 'batterie')}s have no blind cone: in "
                         f"every direction at least one bears at some roll.")
    lines.append("- What bears NOW (and the roll that unmasks the rest): the ARCS lines"
                 + ("; per RANGE CONTROL heading: \"N of M batteries can bear\" = at SOME roll "
                    "(rolling takes time)." if has_guns else "."))
    return "\n".join(lines) + "\n"
