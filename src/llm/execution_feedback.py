"""
Compact engine feedback shared by captains and admirals, independent of recording.

What each ship actually did with its last orders, in sentences a commander
reads at a glance: the helm's applied vs requested throttle, each gun's
current state ("held: P(hit) 0.18 below your 0.30 gate"), commands the
engine rejected/cancelled/expired, and - loudest - a ship that was issued NO
commands at all for one or more checkpoints (it is coasting: that is how two
whole fleets once sat still for a battle while their captains acknowledged
orders). This used to be a raw JSON dump ({"reason":"probability_gate",...}).
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from ..simulation import EVADE_BURN_CONE_DEG
except ImportError:  # pragma: no cover - script-style imports
    from simulation import EVADE_BURN_CONE_DEG

# Weapon transition reasons -> what to tell a commander.
_WEAPON_REASONS = {
    "hold_fire": "HOLD_FIRE (your order)",
    "damaged": "DESTROYED",
    "empty": "out of ammunition",
    "cooldown": "reloading",
    "capacitor_charging": "capacitor charging",
    "heat_limited": "held: heat at the guns-hold line",
    "no_target": "no target designated",
    "target_unavailable": "target gone (destroyed or surrendered)",
    "outside_arc": "target outside its firing arc",
    "no_intercept": "no intercept solution",
    "no_firing_solution": "no firing solution",
    "lead_unavailable": "no lead solution",
    "fired": "firing",
}

# Most recent command problems shown per call.
MAX_PROBLEMS = 6
# How many past decision windows the NO COMMANDS check looks back.
IDLE_LOOKBACK_CHECKPOINTS = 6


def _name(simulation: Any, ship_id: Optional[str]) -> str:
    ship = simulation.get_ship(ship_id) if ship_id and hasattr(simulation, "get_ship") else None
    name = getattr(ship, "name", None) if ship is not None else None
    return f"{name} [{ship_id}]" if name and name != ship_id else str(ship_id)


def _weapon_text(data: Dict[str, Any], target: Optional[str]) -> str:
    reason = data.get("reason")
    if reason == "probability_gate":
        p, gate = data.get("hit_probability"), data.get("min_hit_probability")
        if isinstance(p, (int, float)) and isinstance(gate, (int, float)):
            text = f"held: P(hit) {p:.2f} below your {gate:.2f} gate"
        else:
            text = "held by your hit-probability gate"
    elif reason == "range_gate":
        rng = data.get("max_range_km")
        text = (f"held: target beyond your {rng:.0f} km FIRE_AT_RANGE limit"
                if isinstance(rng, (int, float)) else "held by your range limit")
    else:
        text = _WEAPON_REASONS.get(reason, str(reason))
    if target and reason not in ("hold_fire", "damaged", "empty", "cooldown"):
        text += f" (target {target})"
    return text


# Execution modes of the evasion overlay (simulation.EVASION_OVERLAY_MODES)
# and of EVASIVE: threat-aware evasion has the helm.
_EVASION_MODES = ("EVADE", "TANK", "BRACE", "RUN", "PRESENT", "WOBBLE")


def _helm_text(data: Dict[str, Any]) -> str:
    mode = data.get("mode", "?")
    req, applied = data.get("requested_throttle"), data.get("applied_throttle")
    if mode == "COAST" or req is None:
        return "COASTING (no maneuver)"
    if mode in ("PADLOCK", "MAINTAIN"):
        # The drive is off by design: "requested 100%, applied 0% (turning
        # onto the burn line...)" read as a failed burn.
        return f"{mode}, coasting (engine off" + (", nose on target)" if mode == "PADLOCK" else ")")
    if not req:
        # A throttle-0 order: attitude only - unless its evade overlay has the
        # helm, which burns at full throttle (simulation._evasion_overlay).
        if mode in _EVASION_MODES and isinstance(applied, (int, float)):
            return f"{mode} (evasion has the helm), applied {applied * 100:.0f}% throttle"
        return f"{mode}, throttle 0 (attitude only, no burn)"
    if not isinstance(applied, (int, float)):
        return f"{mode}, requested {req * 100:.0f}% throttle"
    # Why a burn fell short is said by the window line (helm_window_text):
    # "turning onto the burn line, or fuel/engine limited" was printed for
    # evasion holding the drive, a dead drive and the interlock alike.
    text = f"{mode}, requested {req * 100:.0f}% throttle, applied {applied * 100:.0f}%"
    engine = data.get("engine_fraction")
    if isinstance(engine, (int, float)) and engine <= 0.0:
        text += " (DRIVE DEAD)"
    return text


# What each evasion mode does with the drive (simulation._calculate_evasion /
# _evasion_overlay), for the helm window line, on an order that does not burn
# (PADLOCK, MAINTAIN, throttle 0) or the EVADE maneuver. On an order that
# burns the overlay keeps the burn and swings it (execution_state 'swing',
# _SWING_TAG); only RUN leaves the burn line there.
_EVASION_DRIVE = {
    "BRACE": "drive off: a gun round inbound with no time to jink",
    "TANK": "half throttle, turning armour onto a gun round",
    "PRESENT": "token throttle, turning armour onto a terminal torpedo",
    "RUN": "burning away from a guided torpedo",
    "EVADE": "jinking off a gun round at your throttle, off your line",
}
# Short per-mode tags for the helm window line (the full meaning of each mode
# is in the captain doctrine, EVASION MODES).
_EVASION_TAG = {"BRACE": "drive off", "TANK": "half throttle", "PRESENT": "armour to torpedo",
                "RUN": "running", "EVADE": "jinking"}
# The same modes while they swing a burning order's thrust axis inside the
# cone around its burn line (simulation.EVADE_BURN_CONE_DEG): the drive stays
# on the order.
_SWING_TAG = {"EVADE": "dodging", "TANK": "armour", "BRACE": "armour",
              "PRESENT": "armour to torpedo"}
# Base maneuvers whose drive is off by design (the overlay may still burn).
_ENGINE_OFF = ("PADLOCK", "MAINTAIN")
# A window line is printed when the ordered burn flew less than this share.
HELM_SHORTFALL_FRACTION = 0.8


def _span_s(segments: List[Tuple[float, float, Any]], now: float) -> List[Tuple[float, Any]]:
    """[(t, x), ...] change points -> [(seconds, x), ...] up to ``now``."""
    out = []
    for i, (t, _, x) in enumerate(segments):
        end = segments[i + 1][0] if i + 1 < len(segments) else now
        if end > t:
            out.append((end - t, x))
    return out


def helm_window(simulation: Any, ship_id: str, since: float, now: float,
                events: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
    """
    What the helm did with the last window's maneuver, second by second,
    from the engine's own transition log (EXECUTION_STATE, MANEUVER_STARTED,
    DRIVE_INTERLOCK, PROJECTILE_LAUNCHED / TORPEDO_LAUNCHED at this ship) -
    read-only, nothing new is logged. None when there is no window to judge.

    Returns {order: {type, throttle, evade}, window_s, burned_s, modes:
    {mode: s}, swing: {mode: s}, dead_s, damaged: engine fraction or None,
    fuel_s, interlock_s, turning_s, shooters: [names], torpedo_sources:
    [names], turn_left_s, swing_turn_s}. ``modes`` are the seconds evasion
    held the order (an engine-off order, or RUN); ``swing`` the seconds the
    evade overlay swung a burning order's thrust axis - burned on the order,
    and counted in burned_s too; ``swing_turn_s`` the seconds it burned at
    the order's throttle with the nose still turning onto the line (not
    counted as burned - the overlay burns through the whole turn; a tick
    below half the order's throttle there, e.g. a range keeper's
    feed-forward, is turning_s).
    """
    if now - since < 1.0:
        return None
    events = events if events is not None else list(getattr(simulation, "events", []) or [])
    states: List[Tuple[float, int, Dict[str, Any]]] = []
    orders: List[Tuple[float, int, Dict[str, Any]]] = []
    interlock: List[Tuple[float, int, bool]] = []
    shooters: Dict[str, int] = {}
    torpedoes: Dict[str, int] = {}
    for i, event in enumerate(events):
        kind = event.event_type.name
        t = event.timestamp
        if kind in ("PROJECTILE_LAUNCHED", "TORPEDO_LAUNCHED"):
            if event.target_id == ship_id and since - (30.0 if kind == "TORPEDO_LAUNCHED" else 0.0) <= t < now:
                pool = shooters if kind == "PROJECTILE_LAUNCHED" else torpedoes
                pool[event.ship_id] = pool.get(event.ship_id, 0) + 1
            continue
        if event.ship_id != ship_id or t >= now - 1e-9:
            continue
        if kind == "EXECUTION_STATE":
            states.append((t, i, event.data))
        elif kind == "MANEUVER_STARTED":
            orders.append((t, i, event.data))
        elif kind == "DRIVE_INTERLOCK":
            interlock.append((t, i, (event.data or {}).get("phase") == "enter"))

    def clip(points):
        """Change points inside [since, now): the one in force at ``since`` first."""
        before = [p for p in points if p[0] <= since + 1e-9]
        inside = [p for p in points if p[0] > since + 1e-9]
        head = [(since,) + tuple(before[-1][1:])] if before else []
        return head + inside

    states, orders, interlock = clip(states), clip(orders), clip(interlock)
    if not states or not orders:
        return None
    order = orders[-1][2]
    kind = str(order.get("maneuver_type") or "?")
    out: Dict[str, Any] = {
        "order": {"type": kind, "throttle": order.get("throttle"), "evade": order.get("evade")},
        "window_s": now - since, "burned_s": 0.0, "modes": {}, "swing": {}, "swing_turn_s": 0.0,
        "dead_s": 0.0,
        "damaged": None,
        "fuel_s": 0.0, "interlock_s": 0.0, "turning_s": 0.0, "turn_left_s": None,
        "shooters": [n for n, _ in sorted(shooters.items(), key=lambda kv: -kv[1])],
        "torpedo_sources": [n for n, _ in sorted(torpedoes.items(), key=lambda kv: -kv[1])],
    }
    # Merge the three change-point streams into one timeline.
    times = sorted({p[0] for p in states + orders + interlock if p[0] < now})

    def at(points, t):
        cur = None
        for p in points:
            if p[0] <= t + 1e-9:
                cur = p[2]
        return cur

    last_mode = None
    for i, t in enumerate(times):
        end = times[i + 1] if i + 1 < len(times) else now
        dur = end - t
        if dur <= 0:
            continue
        data = at(states, t) or {}
        base = at(orders, t) or {}
        mode = str(data.get("mode") or "?")
        req = data.get("requested_throttle") or 0.0
        app = data.get("applied_throttle") or 0.0
        engine = data.get("engine_fraction")
        fuel = data.get("fuel_fraction")
        base_kind = str(base.get("maneuver_type") or "?")
        burning_order = base_kind not in _ENGINE_OFF and req > 0 and base_kind != "EVASIVE"
        last_mode = mode
        swung = bool(data.get("swing")) and mode in _SWING_TAG
        if mode in _EVASION_DRIVE and not swung:
            out["modes"][mode] = out["modes"].get(mode, 0.0) + dur
            continue
        if swung and data.get("off_line"):
            # Still turning onto the order's line: burning at the order's
            # throttle through the turn (or, for a range keeper's
            # feed-forward, at its share along the correction) - not yet
            # the order's burn either way.
            if app >= 0.5 * req * (engine if isinstance(engine, (int, float)) else 1.0):
                out["swing_turn_s"] += dur
            else:
                out["turning_s"] += dur
            continue
        if swung:
            out["swing"][mode] = out["swing"].get(mode, 0.0) + dur
        if not burning_order:
            continue
        if isinstance(engine, (int, float)) and engine <= 0.0:
            out["dead_s"] += dur
        elif isinstance(fuel, (int, float)) and fuel < 0.99:
            out["fuel_s"] += dur
        elif at(interlock, t):
            out["interlock_s"] += dur
        elif app >= 0.5 * req * (engine if isinstance(engine, (int, float)) else 1.0):
            out["burned_s"] += dur
            if isinstance(engine, (int, float)) and engine < 0.99:
                out["damaged"] = engine
        else:
            out["turning_s"] += dur
    # Still turning at the end of the window: how long until the burn lights.
    heading = order.get("heading_direction")
    if (kind == "HEADING" and isinstance(heading, dict) and last_mode == "HEADING"
            and out["turning_s"] > 0 and hasattr(simulation, "_heading_burn_light_s")):
        ship = simulation.get_ship(ship_id) if hasattr(simulation, "get_ship") else None
        try:
            from ..physics import Vector3D
        except ImportError:  # pragma: no cover - script-style imports
            from physics import Vector3D
        try:
            d = Vector3D(float(heading.get("x", 0)), float(heading.get("y", 0)),
                         float(heading.get("z", 0)))
            if ship is not None and d.magnitude > 0:
                out["turn_left_s"] = float(simulation._heading_burn_light_s(ship, d.normalized()))
        except (AttributeError, TypeError, ValueError):
            pass
    return out


def helm_window_text(simulation: Any, ship_id: str, since: float, now: float,
                     events: Optional[List[Any]] = None) -> Optional[str]:
    """
    One line on what the helm did with the last window's order when the
    order's burn fell short or evasion flew or swung the ship ("" otherwise):
    seconds burned out of the window and, per cause, the seconds it cost -
    evasion holding the drive (which mode, which threat), turning onto the
    burn line (and the turn left), a dead or damaged drive, the drive
    interlock, fuel - and the seconds the evade overlay swung the burn
    inside its cone (still on the order).
    """
    w = helm_window(simulation, ship_id, since, now, events)
    if w is None:
        return None
    order, window = w["order"], w["window_s"]
    kind = order["type"]
    label = "EVADE" if kind == "EVASIVE" else kind
    thr = order.get("throttle")
    burns = kind not in _ENGINE_OFF and kind != "EVASIVE" and isinstance(thr, (int, float)) and thr > 0
    evading = sum(w["modes"].values())
    swinging = sum(w["swing"].values()) + w["swing_turn_s"]
    if (not evading and swinging < 0.5
            and not (burns and w["burned_s"] < HELM_SHORTFALL_FRACTION * window)):
        return None
    head = label + (f" @{thr * 100:.0f}%" if burns else "")
    if order.get("evade"):
        head += f", evade {order['evade']}"
    parts = []
    if burns:
        parts.append(f"burned {w['burned_s']:.0f} of {window:.0f} s on your order")
    names = lambda ids: ", ".join(_name(simulation, s) for s in ids[:2]) + (  # noqa: E731
        f" +{len(ids) - 2}" if len(ids) > 2 else "")
    held = [f"{mode} {w['modes'][mode]:.0f} s {_EVASION_TAG[mode]}"
            for mode in ("BRACE", "TANK", "PRESENT", "RUN", "EVADE")
            if w["modes"].get(mode, 0.0) >= 0.5]
    if held:
        threat = []
        if w["shooters"] and any(m in w["modes"] for m in ("BRACE", "TANK", "EVADE")):
            threat.append(f"fire from {names(w['shooters'])}")
        if w["torpedo_sources"] and any(m in w["modes"] for m in ("PRESENT", "RUN")):
            threat.append(f"torpedoes from {names(w['torpedo_sources'])}")
        parts.append(f"evasion {'held the burn' if burns else 'flew the ship'} {evading:.0f} s: "
                     + ", ".join(held) + (f" ({'; '.join(threat)})" if threat else ""))
    swung = [f"{mode} {w['swing'][mode]:.0f} s {_SWING_TAG[mode]}"
             for mode in ("EVADE", "TANK", "BRACE", "PRESENT")
             if w["swing"].get(mode, 0.0) >= 0.5]
    if swung:
        threat = f" (fire from {names(w['shooters'])})" if w["shooters"] else ""
        parts.append(f"evasion swung the burn within {EVADE_BURN_CONE_DEG:.0f}° of your line "
                     f"{swinging:.0f} s: " + ", ".join(swung) + threat)
    if w["swing_turn_s"] >= 0.5:
        parts.append(f"turning onto the burn line at your throttle under fire "
                     f"{w['swing_turn_s']:.0f} s (evade keeps the drive lit)")
    if w["turning_s"] >= 0.5:
        text = f"turning onto the burn line {w['turning_s']:.0f} s"
        if w["turn_left_s"] is not None and w["turn_left_s"] >= 0.5:
            text += f" (~{w['turn_left_s']:.0f} s of turn still left)"
        parts.append(text)
    if w["dead_s"] >= 0.5:
        parts.append(f"DRIVE DEAD {w['dead_s']:.0f} s (no thrust at all)")
    if w["interlock_s"] >= 0.5:
        parts.append(f"drive interlock {w['interlock_s']:.0f} s (a friendly in your jet)")
    if w["fuel_s"] >= 0.5:
        parts.append(f"fuel-limited {w['fuel_s']:.0f} s")
    if w["damaged"] is not None:
        parts.append(f"drive damaged: thrust {w['damaged'] * 100:.0f}% of rated")
    if not parts:
        return None
    return f"helm T+{since:.0f}-{now:.0f} ({head}): " + "; ".join(parts)


def _command_name(command: Any) -> str:
    if isinstance(command, dict):
        what = command.get("type") or command.get("maneuver_type")
    else:
        what = getattr(command, "maneuver_type", None) or command
    return str(getattr(what, "name", what) or "?")


def idle_checkpoints(events: Iterable[Any], ship_id: str, now: float, interval: float,
                     max_back: int = IDLE_LOOKBACK_CHECKPOINTS) -> List[float]:
    """
    Start times of the consecutive past decision windows, newest first, in
    which this ship was issued no command at all (window k is
    [now - k*interval, now - (k-1)*interval)). Admiral heading orders count
    as commands - they go through the same injection. Windows before the
    battle's first command (the opening coast to the first checkpoint) are
    not decision windows and never count.
    """
    issued = set()
    first_issue = None
    for event in events:
        if event.event_type.name != "COMMAND_STATUS" or event.data.get("status") != "issued":
            continue
        if first_issue is None or event.timestamp < first_issue:
            first_issue = event.timestamp
        if event.ship_id != ship_id or event.timestamp >= now:
            continue
        issued.add(int((now - event.timestamp - 1e-9) // interval) + 1)
    if first_issue is None:
        return []
    starts = []
    for k in range(1, max_back + 1):
        start = now - k * interval
        if start < first_issue - 1e-6 or k in issued:
            break
        starts.append(start)
    return starts


def execution_feedback(simulation: Any, ship_ids: Iterable[str]) -> str:
    """
    ENGINE FEEDBACK block for ``ship_ids`` ("" when there is nothing to say).

    Per ship: a NO COMMANDS ISSUED line when it was given nothing to do, the
    helm's latest state and each gun's; then recent command problems. Ship
    names prefix each line when more than one ship is reported (admiral).
    """
    ids = list(dict.fromkeys(ship_ids))
    idset = set(ids)
    now = float(getattr(simulation, "current_time", 0.0))
    interval = float(getattr(simulation, "decision_interval", 30) or 30)
    since = now - interval
    events = list(getattr(simulation, "events", []) or [])
    latest: Dict[Tuple[str, str], Any] = {}
    problems: List[Any] = []
    for event in events:
        if event.ship_id not in idset:
            continue
        kind = event.event_type.name
        if kind in ("WEAPON_STATUS", "EXECUTION_STATE"):
            latest[(event.ship_id, event.data.get("weapon_slot", "helm"))] = event
        elif (event.timestamp >= since and kind == "COMMAND_STATUS"
              and event.data.get("status") in ("rejected", "cancelled", "expired")):
            problems.append(event)

    multi = len(ids) > 1
    lines: List[str] = []
    for sid in ids:
        prefix = f"{_name(simulation, sid)} " if multi else ""
        ship = simulation.get_ship(sid) if hasattr(simulation, "get_ship") else None
        alive = ship is not None and not getattr(ship, "is_destroyed", False)
        starts = idle_checkpoints(events, sid, now, interval) if alive else []
        if starts:
            n = len(starts)
            when = ", ".join(f"T+{t:.0f}" for t in sorted(starts))
            line = (f"  !! {prefix}NO COMMANDS ISSUED for {n} consecutive checkpoint"
                    f"{'s' if n != 1 else ''} ({when}): nothing was ordered to move, aim or fire")
            speed = getattr(getattr(ship, "velocity", None), "magnitude", None)
            if isinstance(speed, (int, float)):
                line += f"; coasting at {speed / 1000:.2f} km/s"
            lines.append(line + ".")
        helm = latest.get((sid, "helm"))
        window = helm_window_text(simulation, sid, since, now, events) if alive else None
        if window:
            # The window says what the helm did and why; it replaces the
            # last-transition line.
            lines.append(f"  {prefix}{window}")
        elif helm is not None:
            lines.append(f"  {prefix}helm (since T+{helm.timestamp:.0f}): {_helm_text(helm.data)}")
        # Mounts in the same state share a line ("weapon_1, weapon_2 (since
        # T+90): held: ...") - a battleship's four batteries used to print it 4x.
        same: Dict[str, List[str]] = {}
        for (ship_id, slot), event in sorted(latest.items(), key=lambda kv: kv[0][1]):
            if ship_id != sid or slot == "helm":
                continue
            target = _name(simulation, event.target_id) if event.target_id else None
            text = f"(since T+{event.timestamp:.0f}): {_weapon_text(event.data, target)}"
            same.setdefault(text, []).append(slot)
        for text, slots in same.items():
            lines.append(f"  {prefix}{', '.join(slots)} {text}")
    for event in problems[-MAX_PROBLEMS:]:
        prefix = f"{_name(simulation, event.ship_id)} " if multi else ""
        reason = event.data.get("reason") or ""
        lines.append(f"  {prefix}T+{event.timestamp:.0f} command "
                     f"{_command_name(event.data.get('command'))} "
                     f"{str(event.data['status']).upper()}" + (f": {reason}" if reason else ""))
    if not lines:
        return ""
    # What persists (weapon policies) and what expires (maneuvers, queued
    # launches) is in the captain's cached doctrine (YOUR TURN).
    return "ENGINE FEEDBACK (what your ship actually did):\n" + "\n".join(lines)
