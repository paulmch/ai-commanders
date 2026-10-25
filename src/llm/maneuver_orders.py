"""
Maneuver orders: the optional set_maneuver / set_heading parameters, parsed
one way for LLM captains (captain._execute_tool), admirals (issue_order,
applied by battle_runner._apply_admiral_headings) and MCP clients
(mcp_controller._apply_mcp_commands).

  set_maneuver  target_id (name or id), standoff_km + arrival_closing_kps
                (INTERCEPT arrival control), range_km (HOLD_RANGE),
                evade ('full' | 'dodge': the evasion overlay)
  set_heading   toward_point {x, y, z km} | toward_ship (name or id), evade
                (fly there and stop: the engine's range keeping)

Every parameter is optional. With none of them given, callers build exactly
the engine Maneuver they built before these existed (bit-identical seeded
battles), so nothing here is consulted on that path.

Also: the position-vs-direction guard for set_heading. Admirals write
positions ("cluster centre (~90,55,10)") and captains pasted them into
set_heading as a DIRECTION - which flew one corvette 144 degrees away from
the point it was sent to (pass-4 audit CP-03). A direction that is long and
matches coordinates quoted in the orders is still flown as given (the call
is valid), but the captain is told what it did and how to fly to the point.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from ..physics import Vector3D
except ImportError:  # pragma: no cover - script-style imports
    from physics import Vector3D


EVADE_MODES = ("full", "dodge")
# A set_heading direction longer than this is probably a position...
POSITION_LIKE_MAGNITUDE = 2.0
# ...when every component matches a coordinate triple quoted in the orders to
# within this fraction of its value (or 1 km, whichever is larger).
COORD_MATCH_FRACTION = 0.02

_NUMBER = r"[-+]?\d+(?:\.\d+)?"
# "(90, 55, 10)", "(~90,55,10)", "(90 km, 55 km, 10 km)"
_PAREN_TRIPLE = re.compile(
    r"\(\s*~?\s*(" + _NUMBER + r")\s*(?:km)?\s*,\s*~?\s*(" + _NUMBER + r")\s*(?:km)?\s*,"
    r"\s*~?\s*(" + _NUMBER + r")\s*(?:km)?\s*\)", re.IGNORECASE)
# "X=260, Y=0, Z=0", "x: 90 y: 55 z: 10"
_AXIS_TRIPLE = re.compile(
    r"\bx\s*[=:]\s*~?\s*(" + _NUMBER + r")\D{0,8}?\by\s*[=:]\s*~?\s*(" + _NUMBER + r")"
    r"\D{0,8}?\bz\s*[=:]\s*~?\s*(" + _NUMBER + r")", re.IGNORECASE)


class ManeuverArgError(ValueError):
    """An optional maneuver parameter that cannot be flown (message is for the commander)."""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_number(value: Any, name: str) -> Optional[float]:
    """A number from a tool argument ("50", 50, "50 km"); None when absent."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ManeuverArgError(f"{name} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(_NUMBER, str(value))
        if not match:
            raise ManeuverArgError(f"{name} must be a number, got {value!r}")
        number = float(match.group())
    if not math.isfinite(number):
        raise ManeuverArgError(f"{name} must be finite, got {value!r}")
    return number


def parse_evade(value: Any) -> Optional[str]:
    """'full' | 'dodge' | None from a tool argument (true means 'full')."""
    if value is None or value is False:
        return None
    if value is True:
        return "full"
    text = str(value).strip().lower()
    if text in ("", "none", "off", "no", "false", "0"):
        return None
    if text in ("full", "true", "yes", "on", "evade", "1"):
        return "full"
    if text == "dodge":
        return "dodge"
    raise ManeuverArgError(f"evade must be 'full' or 'dodge', got {value!r}")


def parse_point_km(value: Any, name: str = "toward_point") -> Optional[Dict[str, float]]:
    """A world position {x, y, z} in km from a tool argument; None when absent."""
    if value is None or value == {} or value == "":
        return None
    if isinstance(value, dict):
        try:
            point = {k: float(value[k]) for k in ("x", "y", "z")}
        except (KeyError, TypeError, ValueError):
            raise ManeuverArgError(f"{name} needs numeric x, y, z in km, got {value!r}")
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            point = dict(zip(("x", "y", "z"), (float(c) for c in value)))
        except (TypeError, ValueError):
            raise ManeuverArgError(f"{name} needs numeric x, y, z in km, got {value!r}")
    else:
        raise ManeuverArgError(f"{name} needs {{x, y, z}} in km, got {value!r}")
    if not all(math.isfinite(c) for c in point.values()):
        raise ManeuverArgError(f"{name} must be finite, got {value!r}")
    return point


def _ship_names(ship: Any) -> Tuple[str, ...]:
    name = str(getattr(ship, "name", None) or ship.ship_id)
    short = name.split(" ", 1)[1] if " " in name else name   # "TIS Dart-1" -> "Dart-1"
    return name.lower(), short.lower(), str(ship.ship_id).lower()


def _squash(text: str) -> str:
    """'OCS Falchion-1' / 'falchion 1' / 'Falchion_1' -> comparable form."""
    return re.sub(r"[\s_\-]+", "", text)


def resolve_ship(simulation: Any, own_ship_id: str, ref: Any) -> Optional[Any]:
    """
    The ship a commander named (id, full name, name without the hull prefix,
    then a unique partial name - spaces, dashes and underscores ignored, so
    "Falchion 1" finds "OCS Falchion-1"): enemies first, then friendlies.
    None if no live ship matches (or a partial name matches several).
    """
    if ref is None:
        return None
    wanted = str(ref).strip().lower()
    if not wanted:
        return None
    pools = [list(simulation.get_enemy_ships(own_ship_id)),
             list(simulation.get_friendly_ships(own_ship_id))]
    # The label form the turn states teach, "OCS Lancer-2 [beta_2]": the id wins.
    if wanted.endswith("]") and "[" in wanted:
        label_id = wanted[wanted.rindex("[") + 1:-1].strip()
        for pool in pools:
            for ship in pool:
                if str(getattr(ship, "ship_id", "")).lower() == label_id:
                    return ship
        wanted = wanted[:wanted.rindex("[")].strip()
        if not wanted:
            return None
    for pool in pools:
        for ship in pool:
            if wanted in _ship_names(ship):
                return ship
    squashed = _squash(wanted)
    for pool in pools:
        for ship in pool:
            if squashed in {_squash(n) for n in _ship_names(ship)}:
                return ship
    for pool in pools:
        partial = [s for s in pool if squashed and squashed in _squash(_ship_names(s)[0])]
        if len(partial) == 1:
            return partial[0]
    return None


def ship_choices(simulation: Any, own_ship_id: str) -> str:
    """'Name [id], ...' of the ships a maneuver may reference (for errors)."""
    ships = (list(simulation.get_enemy_ships(own_ship_id))
             + list(simulation.get_friendly_ships(own_ship_id)))
    return ", ".join(f"{getattr(s, 'name', s.ship_id)} [{s.ship_id}]" for s in ships) or "(none)"


def _unit_dict(v: Vector3D) -> Optional[Dict[str, float]]:
    if v.magnitude <= 1e-9:
        return None
    u = v.normalized()
    return {"x": round(u.x, 4), "y": round(u.y, 4), "z": round(u.z, 4)}


# ---------------------------------------------------------------------------
# set_maneuver
# ---------------------------------------------------------------------------

def maneuver_extras(
    simulation: Any,
    ship_id: str,
    maneuver_type: Any,
    args: Dict[str, Any],
    target_id: Optional[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Engine Maneuver fields for the optional set_maneuver parameters, plus
    notes on parameters that do not apply to this maneuver type (they are
    ignored, not fatal). Raises ManeuverArgError for values that cannot be
    flown. Empty when none of the parameters are given.
    """
    from ..simulation import ManeuverType

    extras: Dict[str, Any] = {}
    notes: List[str] = []
    standoff = parse_number(args.get("standoff_km"), "standoff_km")
    arrival = parse_number(args.get("arrival_closing_kps"), "arrival_closing_kps")
    keep = parse_number(args.get("range_km"), "range_km")
    evade = parse_evade(args.get("evade"))
    for name, value in (("standoff_km", standoff), ("arrival_closing_kps", arrival),
                        ("range_km", keep)):
        if value is not None and value < 0:
            raise ManeuverArgError(f"{name} must be >= 0, got {value:g}")

    kind = maneuver_type.name
    if maneuver_type == ManeuverType.INTERCEPT:
        if standoff is not None:
            extras["standoff_m"] = standoff * 1000.0
        if arrival is not None:
            extras["arrival_closing_mps"] = arrival * 1000.0
            note = arrival_cap_note(simulation, ship_id, args, standoff, arrival)
            if note:
                notes.append(note)
        if keep is not None:
            notes.append("range_km applies to HOLD_RANGE only and was ignored "
                         "(INTERCEPT stops at standoff_km)")
    elif maneuver_type == ManeuverType.HOLD_RANGE:
        if target_id is None:
            raise ManeuverArgError("HOLD_RANGE needs a target: give target_id "
                                   "(or set a primary target)")
        if keep is None and standoff is not None:
            keep = standoff
            notes.append("HOLD_RANGE took standoff_km as range_km")
        elif standoff is not None:
            notes.append("standoff_km applies to INTERCEPT only and was ignored")
        if arrival is not None:
            notes.append("arrival_closing_kps applies to INTERCEPT only and was ignored")
        if keep is None:
            # Left to the engine: it latches the range at the first tick and a
            # renewed order keeps that range. Filling in the range NOW would
            # re-latch on every renewal - against a closing enemy the hold
            # would bleed range checkpoint after checkpoint.
            note = hold_range_default_note(simulation, ship_id, target_id)
            if note:
                notes.append(note)
        else:
            extras["hold_range_m"] = keep * 1000.0
    else:
        for name, value in (("standoff_km", standoff), ("arrival_closing_kps", arrival),
                            ("range_km", keep)):
            if value is not None:
                notes.append(f"{name} does not apply to {_label(kind)} and was ignored")
    if evade is not None:
        if maneuver_type == ManeuverType.EVASIVE:
            notes.append("evade is implied by EVADE and was ignored")
        else:
            extras["evade"] = evade
    return extras, notes


def hold_range_default_note(simulation: Any, ship_id: str,
                            target_id: Optional[str]) -> Optional[str]:
    """
    What a HOLD_RANGE without range_km will hold: the range it already
    latched (a renewal on the same target), else the range right now.
    """
    from ..simulation import ManeuverType

    ship = _get_ship(simulation, ship_id)
    current = getattr(ship, "current_maneuver", None) if ship is not None else None
    if (current is not None and current.maneuver_type == ManeuverType.HOLD_RANGE
            and current.target_id == target_id and current.hold_range_m is not None):
        return (f"HOLD_RANGE keeps the {current.hold_range_m / 1000.0:.0f} km it latched "
                f"when first ordered (give range_km to change it)")
    now = current_range_m(simulation, ship_id, target_id)
    if now is None:
        return None
    return (f"HOLD_RANGE holds the range when it starts, {now / 1000.0:.0f} km now; "
            f"renewing it keeps that range (give range_km for a set range)")


def arrival_cap_note(simulation: Any, ship_id: str, args: Dict[str, Any],
                     standoff_km: Optional[float], arrival_kps: float) -> Optional[str]:
    """
    The engine flies an INTERCEPT arrival speed no faster than the overshoot
    past the standoff can still be stopped inside it (no fly-through); say so
    when the asked-for speed is over that - or when there is no standoff to
    overshoot into at all.
    """
    if arrival_kps <= 0:
        return None
    if not standoff_km:
        return ("arrival_closing_kps without standoff_km: there is no room past the target "
                "to slow down in, so the ship arrives matched ON the target - give standoff_km "
                "for a range to arrive at")
    try:
        from ..simulation import RANGE_ARRIVAL_ROOM_FRACTION, RANGE_BRAKE_ACCEL_FRACTION
        from ..maneuvers import max_arrival_speed_mps
    except ImportError:  # pragma: no cover - script-style imports
        return None
    ship = _get_ship(simulation, ship_id)
    if ship is None:
        return None
    throttle = parse_number(args.get("throttle"), "throttle")
    throttle = 1.0 if throttle is None else max(0.0, min(1.0, throttle))
    try:
        accel = (ship.kinematic_state.max_acceleration_ms2()
                 * ship.get_effective_thrust_fraction() * throttle)
    except Exception:
        return None
    cap_kps = max_arrival_speed_mps(RANGE_ARRIVAL_ROOM_FRACTION * standoff_km * 1000.0,
                                    RANGE_BRAKE_ACCEL_FRACTION * accel) / 1000.0
    if arrival_kps <= cap_kps + 1e-6:
        return None
    return (f"arrival_closing_kps {arrival_kps:g} is capped to ~{cap_kps:.1f} km/s: faster "
            f"could not be stopped inside the {standoff_km:g} km standoff at your acceleration "
            f"(less if the target burns toward you)")


def _get_ship(simulation: Any, ship_id: str) -> Optional[Any]:
    try:
        return simulation.get_ship(ship_id)
    except Exception:
        return None


def current_range_m(simulation: Any, ship_id: str, target_id: Optional[str]) -> Optional[float]:
    """Range between two ships now (m), or None when either is unknown."""
    if target_id is None:
        return None
    try:
        ship = simulation.get_ship(ship_id)
        target = simulation.get_ship(target_id)
        return round((target.position - ship.position).magnitude, 1)
    except Exception:
        return None


def _label(kind: str) -> str:
    return "EVADE" if kind == "EVASIVE" else kind


# ---------------------------------------------------------------------------
# set_heading
# ---------------------------------------------------------------------------

def heading_target(
    simulation: Any,
    ship_id: str,
    args: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    The live-bearing part of a set_heading call: engine fields for
    toward_ship / toward_point (with heading_direction set to the bearing
    right now), or None when the call gives neither (fly ``direction``).
    toward_ship wins over toward_point wins over direction; notes say what
    was ignored. Raises ManeuverArgError when the ship / point is unusable.
    """
    notes: List[str] = []
    ref = args.get("toward_ship")
    point = parse_point_km(args.get("toward_point"))
    has_ref = ref is not None and str(ref).strip() != ""
    if not has_ref and point is None:
        return None, notes
    has_direction = isinstance(args.get("direction"), dict) and bool(args.get("direction"))
    ship = simulation.get_ship(ship_id)
    fields: Dict[str, Any] = {}
    if has_ref:
        target = resolve_ship(simulation, ship_id, ref)
        if target is None:
            raise ManeuverArgError(
                f"toward_ship: no live ship named {ref!r}. Ships: {ship_choices(simulation, ship_id)}")
        fields["heading_target_id"] = target.ship_id
        aim = target.position
        ignored = [n for n, given in (("toward_point", point is not None),
                                      ("direction", has_direction)) if given]
    else:
        fields["heading_point_km"] = point
        aim = Vector3D(point["x"] * 1000.0, point["y"] * 1000.0, point["z"] * 1000.0)
        ignored = ["direction"] if has_direction else []
    if ignored:
        notes.append(f"give one of direction / toward_point / toward_ship - flew "
                     f"{'toward_ship' if has_ref else 'toward_point'}, ignored {', '.join(ignored)}")
    if ship is not None:
        fields["heading_direction"] = _unit_dict(aim - ship.position)
    return fields, notes


def usable_direction(direction: Any) -> bool:
    """A set_heading direction that can be flown: {x, y, z} numbers, not all zero."""
    if not isinstance(direction, dict) or not direction:
        return False
    try:
        components = [float(direction.get(k, 0.0) or 0.0) for k in ("x", "y", "z")]
    except (TypeError, ValueError):
        return False
    return (all(math.isfinite(c) for c in components)
            and any(abs(c) > 1e-9 for c in components))


def coordinate_triples(texts: Iterable[str]) -> List[Tuple[float, float, float]]:
    """Every (x, y, z) / X=..,Y=..,Z=.. triple quoted in ``texts``."""
    found: List[Tuple[float, float, float]] = []
    for text in texts:
        if not text:
            continue
        for pattern in (_PAREN_TRIPLE, _AXIS_TRIPLE):
            for match in pattern.finditer(str(text)):
                triple = tuple(float(g) for g in match.groups())
                if triple not in found:
                    found.append(triple)  # type: ignore[arg-type]
    return found


def position_like_direction_note(
    direction: Any,
    ship: Any,
    order_texts: Sequence[str],
) -> Optional[str]:
    """
    A note for a set_heading ``direction`` that is really a position quoted
    in the orders (long, and matching a coordinate triple in them), else None.
    The direction is still flown as given; the note explains what that did
    and how to fly to the point instead.
    """
    if not isinstance(direction, dict):
        return None
    try:
        d = Vector3D(float(direction.get("x", 0.0)), float(direction.get("y", 0.0)),
                     float(direction.get("z", 0.0)))
    except (TypeError, ValueError):
        return None
    magnitude = d.magnitude
    if magnitude <= POSITION_LIKE_MAGNITUDE:
        return None
    match = None
    for triple in coordinate_triples(order_texts):
        if all(abs(a - b) <= max(1.0, COORD_MATCH_FRACTION * abs(b))
               for a, b in zip((d.x, d.y, d.z), triple)):
            match = triple
            break
    if match is None:
        return None
    u = d.normalized()
    pt = ", ".join(f"{c:g}" for c in match)
    note = (f"set_heading ACCEPTED as a DIRECTION, not a destination: ({d.x:g}, {d.y:g}, "
            f"{d.z:g}) has length {magnitude:.0f}, so it was flown along the unit direction "
            f"({u.x:+.2f},{u.y:+.2f},{u.z:+.2f}) from wherever you are. Your orders give "
            f"({pt}) as a POSITION")
    position = getattr(ship, "position", None) if ship is not None else None
    if isinstance(position, Vector3D):
        to_point = Vector3D(*(c * 1000.0 for c in match)) - position
        if to_point.magnitude > 1.0:
            off = math.degrees(u.angle_to(to_point.normalized()))
            note += (f": that point is {to_point.magnitude / 1000.0:.0f} km from you, "
                     f"{off:.0f} deg off the direction you flew")
    note += (f". To fly TO it: set_heading toward_point={{x: {match[0]:g}, y: {match[1]:g}, "
             f"z: {match[2]:g}}}.")
    return note


# ---------------------------------------------------------------------------
# MCP set_maneuver
# ---------------------------------------------------------------------------

# Optional set_maneuver parameters an MCP command carries when given.
MCP_MANEUVER_OPTIONS = ("standoff_km", "arrival_closing_kps", "range_km", "evade",
                        "toward_point", "toward_ship")


def mcp_maneuver_check(maneuver_type: str, args: Dict[str, Any]) -> List[str]:
    """
    Validate an MCP set_maneuver call's optional parameters before it is
    queued (the battle state is not at hand here; ships are resolved when the
    command is applied). Returns notes for the tool response; raises
    ManeuverArgError for values that cannot be flown.
    """
    notes: List[str] = []
    for key in ("standoff_km", "arrival_closing_kps", "range_km"):
        value = parse_number(args.get(key), key)
        if value is not None and value < 0:
            raise ManeuverArgError(f"{key} must be >= 0, got {value:g}")
    parse_evade(args.get("evade"))
    parse_point_km(args.get("toward_point"))
    if maneuver_type == "EVASIVE" and args.get("evade"):
        notes.append("evade is implied by EVASIVE and is ignored")
    if maneuver_type != "HEADING" and (args.get("toward_point") or args.get("toward_ship")):
        notes.append("toward_point / toward_ship apply to HEADING only and are ignored")
    direction = args.get("heading_direction")
    if (maneuver_type == "HEADING" and not (args.get("toward_point") or args.get("toward_ship"))
            and not usable_direction(direction)):
        raise ManeuverArgError(
            f"HEADING needs heading_direction {{x, y, z}} (not all zero), toward_point or "
            f"toward_ship - got heading_direction={direction!r}")
    if maneuver_type == "HEADING" and isinstance(direction, dict) and not (
            args.get("toward_point") or args.get("toward_ship")):
        try:
            d = Vector3D(float(direction.get("x", 0.0)), float(direction.get("y", 0.0)),
                         float(direction.get("z", 0.0)))
        except (TypeError, ValueError):
            d = None
        if d is not None and d.magnitude > POSITION_LIKE_MAGNITUDE:
            u = d.normalized()
            notes.append(
                f"heading_direction is a DIRECTION, flown as ({u.x:+.2f},{u.y:+.2f},{u.z:+.2f}) "
                f"from wherever the ship is - if ({d.x:g}, {d.y:g}, {d.z:g}) is a position in "
                f"km, use toward_point instead")
    return notes


def mcp_maneuver_fields(
    simulation: Any,
    ship_id: str,
    maneuver_type: Any,
    params: Dict[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Dict[str, Any], List[str]]:
    """
    (target_id, heading_direction, engine extras, notes) for an MCP
    set_maneuver command that uses any of the optional parameters: names are
    resolved to ids, INTERCEPT / PADLOCK / HOLD_RANGE default to the ship's
    primary target, HEADING takes toward_point / toward_ship. Raises
    ManeuverArgError when the order cannot be flown.
    """
    from ..simulation import ManeuverType

    target_id = params.get("target_id")
    if target_id and simulation.get_ship(target_id) is None:
        target = resolve_ship(simulation, ship_id, target_id)
        if target is None:
            raise ManeuverArgError(
                f"no live ship {target_id!r} for target_id. Ships: "
                f"{ship_choices(simulation, ship_id)}")
        target_id = target.ship_id
    if not target_id and maneuver_type in (ManeuverType.INTERCEPT, ManeuverType.PADLOCK,
                                           ManeuverType.HOLD_RANGE):
        ship = simulation.get_ship(ship_id)
        target_id = getattr(ship, "primary_target_id", None)
        if not target_id:
            raise ManeuverArgError(f"{maneuver_type.name} needs target_id (no primary target set)")
    heading_direction = params.get("heading_direction")
    if maneuver_type == ManeuverType.HEADING:
        live, notes = heading_target(simulation, ship_id, params)
        extras: Dict[str, Any] = {}
        if live is not None:
            extras.update(live)
            heading_direction = extras.pop("heading_direction", None)
        evade = parse_evade(params.get("evade"))
        if evade is not None:
            extras["evade"] = evade
        return target_id, heading_direction, extras, notes
    extras, notes = maneuver_extras(simulation, ship_id, maneuver_type, params, target_id)
    return target_id, heading_direction, extras, notes


def fleet_heading_vector(simulation: Any, ships: Sequence[Any],
                         params: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    The ONE heading a fleet flies for an "all ships" HEADING order: the given
    direction, or the bearing from the fleet centre to toward_point /
    toward_ship now (re-bearing every hull on its own would collapse the
    formation onto one point). None when nothing usable is given.
    """
    live = [s for s in ships if not s.is_destroyed and not getattr(s, "is_dying", False)]
    if not live:
        return None
    n = float(len(live))
    centre = Vector3D(sum(s.position.x for s in live) / n, sum(s.position.y for s in live) / n,
                      sum(s.position.z for s in live) / n)
    ref = params.get("toward_ship")
    if ref is not None and str(ref).strip():
        target = resolve_ship(simulation, live[0].ship_id, ref)
        if target is None:
            raise ManeuverArgError(
                f"toward_ship: no live ship named {ref!r}. Ships: "
                f"{ship_choices(simulation, live[0].ship_id)}")
        return _unit_dict(target.position - centre)
    point = parse_point_km(params.get("toward_point"))
    if point is not None:
        return _unit_dict(Vector3D(point["x"] * 1000.0, point["y"] * 1000.0,
                                   point["z"] * 1000.0) - centre)
    direction = params.get("heading_direction")
    return direction if isinstance(direction, dict) and direction else None


# ---------------------------------------------------------------------------
# Admiral orders
# ---------------------------------------------------------------------------

# issue_order maneuver names (HEADING is implied by heading/toward_*).
ADMIRAL_MANEUVERS = ("INTERCEPT", "HOLD_RANGE", "EVADE", "BRAKE", "MAINTAIN", "PADLOCK", "HEADING")


def _quiet(parse, *a) -> Any:
    """Admirals get no tool feedback: an unusable optional value is dropped."""
    try:
        return parse(*a)
    except ManeuverArgError:
        return None


def admiral_maneuver_spec(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    The structured maneuver of an issue_order call (maneuver, maneuver_target,
    standoff_km, arrival_closing_kps, range_km, toward_point, toward_ship,
    evade), normalised, or None when the call has none - a plain heading
    order stays exactly what it was. HEADING specs carry toward_point /
    toward_ship / evade for a heading order.
    """
    kind = str(args.get("maneuver") or "").strip().upper()
    if kind == "EVASIVE":
        kind = "EVADE"
    if kind and kind not in ADMIRAL_MANEUVERS:
        kind = ""
    point = _quiet(parse_point_km, args.get("toward_point"))
    ref = args.get("toward_ship")
    ref = str(ref).strip() if ref is not None and str(ref).strip() else None
    evade = _quiet(parse_evade, args.get("evade"))
    if not kind:
        if point is None and ref is None and evade is None:
            return None
        if point is None and ref is None and not isinstance(args.get("heading"), dict):
            return None     # evade with nothing to fly
        kind = "HEADING"
    spec: Dict[str, Any] = {"maneuver_type": kind}
    if kind == "HEADING":
        if ref is not None:
            spec["toward_ship"] = ref
        elif point is not None:
            spec["toward_point"] = point
    elif kind in ("INTERCEPT", "HOLD_RANGE", "PADLOCK"):
        # An explicit maneuver_target must resolve (else the order is not
        # applied and the captain is told); the gun target is only a hint -
        # it is often prose ("the nearest destroyer").
        if args.get("maneuver_target"):
            spec["target"] = str(args["maneuver_target"])
        elif args.get("suggested_target"):
            spec["target_hint"] = str(args["suggested_target"])
    if kind != "HEADING":
        for key in ("standoff_km", "arrival_closing_kps", "range_km"):
            value = _quiet(parse_number, args.get(key), key)
            if value is not None and value >= 0:
                spec[key] = value
    if evade is not None and kind != "EVADE":
        spec["evade"] = evade
    throttle = _quiet(parse_number, args.get("throttle"), "throttle")
    spec["throttle"] = 1.0 if throttle is None else max(0.0, min(1.0, throttle))
    return spec


def admiral_spec_maneuver(simulation: Any, ship: Any, spec: Dict[str, Any],
                          throttle: float) -> Tuple[Any, List[str]]:
    """
    The engine Maneuver for an admiral's structured (non-formation) order on
    ``ship``. INTERCEPT / HOLD_RANGE / PADLOCK fly relative to the named
    target, else the ship's primary target, else the nearest enemy. Raises
    ManeuverArgError when it cannot be flown (e.g. no such ship).
    """
    from ..simulation import Maneuver, ManeuverType

    kind = spec.get("maneuver_type", "MAINTAIN")
    maneuver_type = ManeuverType["EVASIVE" if kind == "EVADE" else kind]
    args = {k: spec.get(k) for k in ("standoff_km", "arrival_closing_kps", "range_km",
                                     "evade", "toward_point", "toward_ship")}
    target_id = None
    heading_direction = None
    if maneuver_type == ManeuverType.HEADING:
        live, notes = heading_target(simulation, ship.ship_id, args)
        extras: Dict[str, Any] = dict(live or {})
        heading_direction = extras.pop("heading_direction", None)
        if args.get("evade"):
            extras["evade"] = parse_evade(args["evade"])
    else:
        if maneuver_type in (ManeuverType.INTERCEPT, ManeuverType.PADLOCK,
                             ManeuverType.HOLD_RANGE):
            target = resolve_ship(simulation, ship.ship_id, spec.get("target"))
            if target is None and spec.get("target"):
                raise ManeuverArgError(
                    f"no live ship named {spec['target']!r} (ships: "
                    f"{ship_choices(simulation, ship.ship_id)})")
            if target is None and spec.get("target_hint"):
                # The order's gun target, if it names a ship; else fall back.
                target = resolve_ship(simulation, ship.ship_id, spec["target_hint"])
            if target is None:
                target_id = getattr(ship, "primary_target_id", None)
                if not target_id:
                    enemies = simulation.get_enemy_ships(ship.ship_id)
                    if enemies:
                        target_id = min(enemies, key=lambda e: (e.position - ship.position)
                                        .magnitude).ship_id
            else:
                target_id = target.ship_id
            if target_id is None:
                raise ManeuverArgError(f"{kind} has no target")
        extras, notes = maneuver_extras(simulation, ship.ship_id, maneuver_type, args, target_id)
    return Maneuver(
        maneuver_type=maneuver_type,
        target_id=target_id,
        start_time=simulation.current_time,
        duration=getattr(simulation, "decision_interval", 30.0),
        throttle=throttle,
        heading_direction=heading_direction,
        **extras,
    ), notes


def apply_admiral_spec(simulation: Any, ship: Any, order: Any, spec: Dict[str, Any],
                       throttle: float) -> bool:
    """
    Fly one admiral structured maneuver order (AdmiralOrder.maneuver) on
    ``ship`` (battle_runner._apply_admiral_headings). On success the order
    records what was flown (maneuver_applied, naming the ship it resolved
    to) and its applied_throttle; when it cannot be flown (no such ship,
    nothing to steer at) it records maneuver_error, the ship keeps its
    previous maneuver and the captain is told the order was NOT applied
    (admiral_maneuver_lines). Returns whether it was applied.
    """
    from ..simulation import ManeuverType

    order.maneuver_error = None
    order.maneuver_applied = None
    try:
        maneuver, _notes = admiral_spec_maneuver(simulation, ship, spec, throttle)
        if maneuver.maneuver_type == ManeuverType.HEADING and not maneuver.live_heading:
            if maneuver.heading_direction is None and isinstance(
                    getattr(order, "heading", None), dict):
                maneuver.heading_direction = dict(order.heading)
            if not usable_direction(maneuver.heading_direction):
                raise ManeuverArgError("no heading to fly")
    except (ManeuverArgError, KeyError, ValueError) as e:
        order.maneuver_error = str(e) or type(e).__name__
        return False
    if not simulation.inject_command(ship.ship_id, maneuver):
        order.maneuver_error = "the ship cannot maneuver"
        return False
    order.applied_throttle = throttle
    target = simulation.get_ship(maneuver.target_id) if maneuver.target_id else None
    order.maneuver_applied = describe_maneuver_spec(
        spec, getattr(target, "name", None) if target is not None else None)
    return True


def describe_maneuver_spec(spec: Optional[Dict[str, Any]],
                           target_name: Optional[str] = None) -> Optional[str]:
    """
    One line for an admiral's structured maneuver order (AdmiralOrder.maneuver),
    e.g. "INTERCEPT Dart-2, standoff 50 km, evade dodge" - for the captain's
    order rendering and logs. ``target_name``: the ship it was actually flown
    against (overrides the spec's). None when there is no spec.
    """
    if not spec:
        return None
    kind = str(spec.get("maneuver_type", "?"))
    parts = [_label(kind)]
    target = target_name or spec.get("target")
    if target:
        parts[0] += f" {target}"
    if spec.get("toward_ship"):
        parts[0] += f" to ship {spec['toward_ship']} (joins it, matching its velocity)"
    elif spec.get("toward_point"):
        p = spec["toward_point"]
        parts[0] += (f" to point ({p.get('x', 0):g}, {p.get('y', 0):g}, {p.get('z', 0):g}) km "
                     f"(flies there and stops)")
    if spec.get("standoff_km") is not None:
        parts.append(f"standoff {spec['standoff_km']:g} km")
    if spec.get("arrival_closing_kps") is not None:
        parts.append(f"arrive closing <= {spec['arrival_closing_kps']:g} km/s")
    if spec.get("range_km") is not None:
        parts.append(f"hold {spec['range_km']:g} km")
    if spec.get("throttle") is not None:
        parts.append(f"{float(spec['throttle']) * 100:.0f}% throttle")
    if spec.get("evade"):
        parts.append(f"evade {spec['evade']}")
    return ", ".join(parts)


def admiral_order_moves_ship(order: Any) -> bool:
    """
    True when the runner flies this admiral order on the ship: a heading, or
    a structured maneuver that was applied (not one marked maneuver_error).
    """
    if getattr(order, "maneuver_error", None):
        return False
    return (isinstance(getattr(order, "heading", None), dict)
            or bool(getattr(order, "maneuver", None)))


def admiral_maneuver_lines(order: Any) -> List[str]:
    """
    The captain-facing lines for an admiral order's structured maneuver
    (AdmiralOrder.maneuver, applied by battle_runner._apply_admiral_headings):
    ALREADY APPLIED (keep it), or NOT APPLIED with the reason (the ship is
    still flying whatever it flew before - choose a maneuver). A plain
    heading order is rendered by its own HEADING ORDER line; only its evade
    overlay is added here.
    """
    spec = getattr(order, "maneuver", None)
    if not spec:
        return []
    error = getattr(order, "maneuver_error", None)
    if error:
        return [f"      >>> MANEUVER ORDER NOT APPLIED ({describe_maneuver_spec(spec)}): "
                f"{error}. Your ship is still flying its previous maneuver - choose your own "
                f"(set_maneuver/set_heading)."]
    if isinstance(getattr(order, "heading", None), dict):
        if spec.get("evade"):
            return [f"      >>> ...with evade {spec['evade']}: evasion flies the ship only "
                    f"while a torpedo or slug is inbound, then the heading resumes."]
        return []
    what = getattr(order, "maneuver_applied", None) or describe_maneuver_spec(spec)
    return [f"      >>> MANEUVER ORDER (ALREADY APPLIED to your ship): {what}. Keep it: "
            f"calling set_maneuver/set_heading replaces it. Override only for a reason."]
