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
    text = f"{mode}, requested {req * 100:.0f}% throttle, applied {applied * 100:.0f}%"
    if applied < 0.5 * req:
        text += " (turning onto the burn line, or fuel/engine limited)"
    return text


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
        if helm is not None:
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
