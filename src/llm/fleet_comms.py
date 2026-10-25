"""
Fleet comms engine-side helpers: reports and compliance, the datalink
auto-signals, the per-ship inbox budget, and the battle context handed to an
admiral answering a captain's question.

Everything here reads simulation state and never writes it: no RNG draws, no
command injection, so a seeded battle is bit-identical with or without comms.
The message model and routing live in communication.py; the runner
(battle_runner.py) wires these into the checkpoint loop.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .communication import (
    CHANNEL_ADMIRAL_TO_CAPTAIN,
    CHANNEL_DIRECT,
    CHANNEL_ENEMY,
    CHANNEL_FLEET_NET,
    INBOX_MAX_ENEMY_ITEMS,
    INBOX_MAX_ITEMS,
    REPORT_INTENT_CHARS,
    REPORT_REASON_CHARS,
    REQUEST_TEXT_CHARS,
    ROLE_ENGINE,
    CommsMessage,
    FleetCommunicationChannel,
    clip,
)

# Engine datalink signal kinds (sender_role 'engine').
SIGNAL_LAUNCH = "LAUNCH"
SIGNAL_KILL = "KILL"
SIGNAL_LOSS = "LOSS"
SIGNAL_PD_LOSS = "PD_LOSS"

DATALINK_NAME = "FLEET DATALINK"

# Inbox priority: LLM-authored traffic first (direct > admiral > fleet net >
# enemy channel), then the datalink (deaths > PD losses > launches).
_ENGINE_PRIORITY = {SIGNAL_KILL: 0, SIGNAL_LOSS: 0, SIGNAL_PD_LOSS: 1, SIGNAL_LAUNCH: 2}
_CHANNEL_PRIORITY = {CHANNEL_DIRECT: 0, CHANNEL_ADMIRAL_TO_CAPTAIN: 1, CHANNEL_FLEET_NET: 2,
                     CHANNEL_ENEMY: 3}

# Maneuver words an admiral's prose order can name, mapped to the
# captain-facing maneuver names (EVASIVE is the engine's name for EVADE).
_MANEUVER_WORDS = {
    "INTERCEPT": "INTERCEPT", "EVADE": "EVADE", "EVASIVE": "EVADE", "BRAKE": "BRAKE",
    "PADLOCK": "PADLOCK", "MAINTAIN": "MAINTAIN", "COAST": "MAINTAIN",
    "HOLD_RANGE": "HOLD_RANGE", "HOLD RANGE": "HOLD_RANGE", "HEADING": "HEADING",
}
# Lower-case prose is ambiguous ("maintain formation", "heading toward") - only
# these verbs are trusted when the admiral did not write a maneuver in caps.
_LOWERCASE_MANEUVER_WORDS = ("intercept", "evade", "brake", "padlock")
_UPPER_RE = re.compile(r"\b(" + "|".join(sorted(
    (re.escape(w) for w in _MANEUVER_WORDS), key=len, reverse=True)) + r")\b")
_LOWER_RE = re.compile(r"\b(" + "|".join(_LOWERCASE_MANEUVER_WORDS) + r")\b", re.IGNORECASE)
# "INTERCEPT (NOT PADLOCK)", "do NOT padlock yet", "never EVADE": a maneuver
# word right after a negation is what the admiral ruled out, not ordered.
_NEGATION_RE = re.compile(r"\b(not|no|never|don'?t|avoid|stop|cease)\W+(\w+\W+)?$", re.IGNORECASE)
# An order steering to coordinates - "(260,0,0)", "(X=260, Y=0)", "(~90,55)" -
# can only be flown with set_heading: INTERCEPT takes a ship, not a point.
_COORD = r"\s*(?:[xyz]\s*[=:~]?\s*)?~?\s*[-+]?\d+(?:\.\d+)?\s*(?:km)?\s*"
_POINT_RE = re.compile(r"\(" + _COORD + "," + _COORD + r"[,)]", re.IGNORECASE)


def maneuver_name(maneuver_type: Any) -> str:
    """Captain-facing maneuver name for an engine ManeuverType (or its name)."""
    name = getattr(maneuver_type, "name", None) or str(maneuver_type or "")
    return "EVADE" if name == "EVASIVE" else name


def ship_alive(ship: Any) -> bool:
    """In the fight: exists, not destroyed, not a dying hulk, not surrendered."""
    return bool(ship) and not ship.is_destroyed and not getattr(ship, "is_dying", False) \
        and not getattr(ship, "is_surrendered", False)


def pd_operational(ship: Any) -> Tuple[int, int]:
    """(operational, total) point-defense turrets."""
    turrets = getattr(ship, "point_defense", None) or []
    return sum(1 for pd in turrets if getattr(pd, "is_operational", True)), len(turrets)


# ---------------------------------------------------------------------------
# Reports and compliance
# ---------------------------------------------------------------------------

def ordered_maneuvers(order: Any) -> List[str]:
    """
    Maneuvers an admiral order asks for, in the order the text names them.

    A structured maneuver on the order (if the order schema carries one) or a
    structured heading wins; otherwise the prose is scanned - maneuver names
    in capitals first, then the unambiguous verbs in any case. A name right
    after a negation ("NOT PADLOCK", "do not evade") is not an order. Prose
    often names several ("INTERCEPT ...; EVADE if torpedoes come inbound"):
    flying any one of them complies.
    """
    if order is None:
        return []
    structured = getattr(order, "maneuver", None) or getattr(order, "maneuver_type", None)
    if isinstance(structured, dict):
        structured = structured.get("maneuver_type") or structured.get("type")
    if structured:
        return [maneuver_name(str(structured).upper())]
    if isinstance(getattr(order, "heading", None), dict):
        return ["HEADING"]
    text = str(getattr(order, "order_text", "") or "")

    def asked(matches: Iterable[Any]) -> List[str]:
        return [_MANEUVER_WORDS[m.group(1).upper()] for m in matches
                if not _NEGATION_RE.search(text[max(0, m.start() - 24):m.start()])]

    found = asked(_UPPER_RE.finditer(text))
    if not found:
        found = asked(_LOWER_RE.finditer(text))
    seen: List[str] = []
    for name in found:
        if name not in seen:
            seen.append(name)
    return seen


def primary_order(orders: Iterable[Any]) -> Any:
    """
    The admiral's real order among a ship's orders this checkpoint.

    The runner appends synthetic [CLARIFICATION]/[CRITICAL] orders to the
    same list during a discussion; the admiral's own order comes first.
    """
    for order in orders or []:
        text = str(getattr(order, "order_text", "") or "")
        if not text.startswith(("[CLARIFICATION]", "[CRITICAL]")):
            return order
    return None


def executed_maneuver(ship: Any, commands: Iterable[Any], now: float) -> str:
    """
    The maneuver the ship flies after this decision.

    The captain's own maneuver command wins (it overrides an admiral HEADING
    applied before the captains decided); otherwise the HEADING the runner
    already injected this checkpoint; otherwise none was issued (the previous
    maneuver runs out and the ship coasts).
    """
    for cmd in commands or []:
        if hasattr(cmd, "maneuver_type") and hasattr(cmd, "throttle"):
            return maneuver_name(cmd.maneuver_type)
    current = getattr(ship, "current_maneuver", None)
    if current is not None and abs(getattr(current, "start_time", -1.0) - now) < 1e-6:
        return maneuver_name(current.maneuver_type)
    return "NONE"


def commanded_target(ship: Any, commands: Iterable[Any]) -> Optional[str]:
    """Target id after this decision: a set_target command, else the current one."""
    for cmd in commands or []:
        if isinstance(cmd, dict) and cmd.get("type") == "set_target":
            return cmd.get("target_id")
    return getattr(ship, "primary_target_id", None)


def committed_rounds(commands: Iterable[Any]) -> int:
    """Torpedo rounds the captain committed this decision."""
    return sum(1 for cmd in commands or []
               if isinstance(cmd, dict) and cmd.get("type") == "launch_torpedo")


def compliance(
    order: Any,
    channel: FleetCommunicationChannel,
    executed: str,
    actual_target_id: Optional[str],
    launched: int,
    final: bool,
) -> Optional[Dict[str, Any]]:
    """
    Ordered vs executed for one ship (None when the ship had no order).

    Plan schema keys {ordered_maneuver, executed_maneuver, ordered_target,
    actual_target, ordered_salvo, launched} plus `final` (False at filing:
    `launched` counts the rounds the captain committed; True in the Admiral's
    copy at the next checkpoint: the rounds that actually left the tubes),
    `deviations` - one short line per mismatch, empty when the ship complied -
    and `followed` (no deviations). `deviations`/`followed` are the verdict.

    `ordered_maneuver` stays a single comparable value for consumers that
    test it against `executed_maneuver`: an order naming several maneuvers
    ("INTERCEPT ...; EVADE if torpedoes come inbound", all of them in
    `ordered_maneuvers`) records the one the ship flew when it flew one of
    them, and the "/"-joined options when it flew none - so equal means
    complied and different means deviated, as `deviations` says.
    """
    if order is None:
        return None
    ordered = ordered_maneuvers(order)
    accepted = list(ordered)
    if ordered and "HEADING" not in accepted and _POINT_RE.search(str(getattr(order, "order_text", "") or "")):
        accepted.append("HEADING")  # "INTERCEPT toward (260,0,0)" flown as set_heading to the point
    flew_ordered = not ordered or executed in accepted
    raw_target = getattr(order, "torpedo_target", None) or getattr(order, "suggested_target", None)
    ordered_target_id = None
    if raw_target:
        faction = channel.get_faction(getattr(order, "target_ship_id", "") or "")
        matches = channel.match_ship(
            str(raw_target), prefer=lambda sid: channel.get_faction(sid) != faction)
        ordered_target_id = matches[0] if len(matches) == 1 else None
    ordered_target = channel.ship_name(ordered_target_id) if ordered_target_id else (raw_target or None)
    actual_target = channel.ship_name(actual_target_id) if actual_target_id else None
    salvo = getattr(order, "torpedo_salvo", None)
    try:
        salvo = None if salvo is None else int(salvo)
    except (TypeError, ValueError):
        salvo = None

    deviations: List[str] = []
    if not flew_ordered:
        deviations.append(f"maneuver: ordered {'/'.join(ordered)}, flew {executed}")
    if ordered_target_id and actual_target_id != ordered_target_id:
        deviations.append(f"target: ordered {ordered_target}, engaging {actual_target or 'nothing'}")
    verb = "launched" if final else "committed"
    if salvo is not None and salvo > 0 and launched < salvo:
        deviations.append(f"salvo: ordered {salvo}, {verb} {launched}")
    elif salvo == 0 and launched > 0:
        deviations.append(f"salvo: ordered hold, {verb} {launched}")
    return {
        "ordered_maneuver": (executed if ordered and flew_ordered else "/".join(ordered)) or None,
        "ordered_maneuvers": ordered,
        "executed_maneuver": executed,
        "ordered_target": ordered_target,
        "actual_target": actual_target,
        "ordered_salvo": salvo,
        "launched": launched,
        "final": final,
        "followed": not deviations,
        "deviations": deviations,
    }


def report_text(report: Optional[Dict[str, Any]]) -> Optional[str]:
    """One-line form of a captain report (captain_decision.acknowledgment)."""
    if not report:
        return None
    status = report.get("status", "ACK")
    parts = [status + (" (implicit)" if report.get("implicit") else "")]
    if report.get("reason"):
        parts.append(report["reason"])
    if report.get("intent"):
        parts.append(f"intent: {report['intent']}")
    contact = report.get("contact")
    if contact:
        parts.append(f"contact: {contact.get('ship')} {contact.get('assessment')}"
                     + (f" ({contact['note']})" if contact.get("note") else ""))
    request = report.get("request")
    if request:
        parts.append(f"request: {request.get('kind')}"
                     + (f" - {request['text']}" if request.get("text") else ""))
    return " | ".join(parts)


def fold_admiral_signals(report: Dict[str, Any], signals: List[Dict[str, Any]]) -> None:
    """
    File the ship's ADMIRAL-addressed fleet_signals on its report.

    The Admiral's one input channel is captain_reports, and its CAPTAIN
    REPORTS view collapses a report with status ACK, no deviation, no contact
    and no request into "ACK, nothing to flag". So each signal is filed in a
    field that view always shows, never only under `signals` (kept verbatim):
    - BREAKING_OFF makes an ACK a DEVIATE (the captain is leaving its
      orders), its text the reason when the captain gave none;
    - PD_COVER_REQUEST is a PD_COVER request;
    - TARGET_CLAIM / TOT_SYNC / FREE fill the request slot with the signal's
      own kind (the Admiral reads "asks [TOT_SYNC]: ..."), the next one an
      empty intent.
    Slots the captain's own report_to_admiral already filled are kept.
    """
    if not signals:
        return
    report.setdefault("signals", []).extend(signals)
    for sig in signals:
        kind = sig.get("kind") or "FREE"
        text = str(sig.get("text") or "")
        head = kind + (f" {sig['target']}" if sig.get("target") else "")
        if sig.get("eta_s") is not None:
            head += f" ETA {sig['eta_s']:.0f}s"
        if kind == "BREAKING_OFF":
            if report.get("status", "ACK") == "ACK":
                report["status"] = "DEVIATE"
                report["implicit"] = False
            if not report.get("reason"):
                report["reason"] = clip(f"[{head}] {text}", REPORT_REASON_CHARS)
                continue
        elif not report.get("request"):
            if kind == "PD_COVER_REQUEST":
                report["request"] = {"kind": "PD_COVER", "text": clip(text, REQUEST_TEXT_CHARS)}
            else:
                lead = head[len(kind):].strip()
                report["request"] = {"kind": kind,
                                     "text": clip(f"{lead} - {text}" if lead else text, REQUEST_TEXT_CHARS)}
            continue
        if not report.get("intent"):
            report["intent"] = clip(f"[{head}] {text}", REPORT_INTENT_CHARS)


# ---------------------------------------------------------------------------
# Engine datalink: auto-signals without an LLM call
# ---------------------------------------------------------------------------

def torpedo_eta(torp_flight: Any, target: Any) -> Optional[float]:
    """Seconds to impact of one round on its target, None when not closing."""
    torp = torp_flight.torpedo
    offset = torp.position - target.position
    dist = offset.magnitude
    if dist <= 0:
        return 0.0
    closing = -(torp.velocity - target.velocity).dot(offset.normalized())
    return dist / closing if closing > 0 else None


def torpedo_etas(simulation: Any, source_id: str, target_id: str) -> List[float]:
    """Seconds to impact of the source's live, closing rounds on the target."""
    target = simulation.get_ship(target_id)
    if target is None:
        return []
    etas = (torpedo_eta(tf, target) for tf in getattr(simulation, "torpedoes", None) or []
            if tf.source_ship_id == source_id and tf.torpedo.target_id == target_id)
    return [eta for eta in etas if eta is not None]


@dataclass
class EngineDatalink:
    """
    Turns the last interval's simulation events into fleet-net signals.

    Friendly torpedo launches (target + live ETA), kills and losses, and PD
    turrets lost are what a captain needs to coordinate without spending a
    tool call on it. One signal per (launcher, target) pair, per death and
    per ship that lost PD keeps the inbox budget for what LLMs wrote.
    """
    cursor: int = 0
    announced_deaths: set = field(default_factory=set)
    pd_counts: Dict[str, int] = field(default_factory=dict)

    def prime(self, simulation: Any) -> None:
        """Start the event window and the PD baseline (battle setup)."""
        self.cursor = len(getattr(simulation, "events", []) or [])
        self.pd_counts = {sid: pd_operational(ship)[0]
                          for sid, ship in getattr(simulation, "ships", {}).items()}

    def collect(
        self,
        simulation: Any,
        channel: FleetCommunicationChannel,
        factions: Iterable[str],
        checkpoint: int,
        now: float,
        msg_id: Callable[[str], str],
    ) -> List[CommsMessage]:
        """Signals for `factions` from the events since the previous call."""
        factions = set(factions)
        events = list(getattr(simulation, "events", []) or [])[self.cursor:]
        self.cursor += len(events)

        def alive_ids(faction: str, exclude: Optional[str] = None) -> List[str]:
            return [sid for sid in channel.faction_ship_ids(faction)
                    if sid != exclude and ship_alive(simulation.get_ship(sid))]

        def make(faction: str, kind: str, text: str, recipients: List[str],
                 structured: Dict[str, Any]) -> Optional[CommsMessage]:
            if faction not in factions or not recipients:
                return None
            return CommsMessage(
                msg_id=msg_id(f"engine:{faction}"), channel=CHANNEL_FLEET_NET, kind=kind,
                sender_id=f"engine:{faction}", sender_role=ROLE_ENGINE, sender_name=DATALINK_NAME,
                faction=faction, recipient_spec="ALL", recipient_ids=recipients, text=text,
                structured=structured, checkpoint=checkpoint, sent_t=now, deliver_at_t=now)

        launches: Dict[Tuple[str, str], int] = {}
        deaths: List[Tuple[str, Optional[str]]] = []
        for event in events:
            kind = event.event_type.name
            if kind == "TORPEDO_LAUNCHED" and event.ship_id and event.target_id:
                key = (event.ship_id, event.target_id)
                launches[key] = launches.get(key, 0) + 1
            elif kind in ("SHIP_DYING", "SHIP_DESTROYED") and event.ship_id:
                if event.ship_id not in self.announced_deaths:
                    self.announced_deaths.add(event.ship_id)
                    deaths.append((event.ship_id, (event.data or {}).get("killer_id")))

        out: List[CommsMessage] = []
        for (src, tgt), rounds in launches.items():
            faction = channel.get_faction(src)
            if not faction:
                continue
            etas = torpedo_etas(simulation, src, tgt)
            src_name, tgt_name = channel.ship_name(src), channel.ship_name(tgt)
            if etas:
                lo, hi = min(etas), max(etas)
                eta = f"ETA {lo:.0f}s" if hi - lo < 1.5 else f"ETA {lo:.0f}-{hi:.0f}s"
                tail = f", {len(etas)} in flight {eta}"
            else:
                tail = ", none still in flight"
            text = f"{src_name} launched {rounds} torpedo{'es' if rounds != 1 else ''} at {tgt_name}{tail}"
            msg = make(faction, SIGNAL_LAUNCH, text, alive_ids(faction, exclude=src), {
                "subject_id": src, "subject_name": src_name, "target_id": tgt,
                "target_name": tgt_name, "rounds": rounds, "in_flight": len(etas),
                "eta_s_min": round(min(etas), 1) if etas else None,
                "eta_s_max": round(max(etas), 1) if etas else None,
            })
            if msg:
                out.append(msg)

        for dead, killer in deaths:
            dead_faction = channel.get_faction(dead)
            if not dead_faction:
                continue
            dead_name = channel.ship_name(dead)
            killer_name = channel.ship_name(killer) if killer else None
            for faction in sorted({f for (_, _, f) in channel.ships.values()}):
                own = faction == dead_faction
                text = (f"{dead_name} {'LOST' if own else 'DESTROYED'}"
                        + (f" (by {killer_name})" if killer_name and not own else ""))
                msg = make(faction, SIGNAL_LOSS if own else SIGNAL_KILL, text, alive_ids(faction), {
                    "subject_id": dead, "subject_name": dead_name, "killer_id": killer,
                    "killer_name": killer_name})
                if msg:
                    out.append(msg)

        for sid, (_, _, faction) in channel.ships.items():
            ship = simulation.get_ship(sid)
            if ship is None:
                continue
            ready, total = pd_operational(ship)
            before = self.pd_counts.get(sid, ready)
            self.pd_counts[sid] = ready
            if ready >= before or not ship_alive(ship):
                continue
            name = channel.ship_name(sid)
            msg = make(faction, SIGNAL_PD_LOSS,
                       f"{name} lost {before - ready} PD turret{'s' if before - ready != 1 else ''} "
                       f"({ready}/{total} left)",
                       alive_ids(faction, exclude=sid),
                       {"subject_id": sid, "subject_name": name, "pd_operational": ready,
                        "pd_total": total, "pd_lost": before - ready})
            if msg:
                out.append(msg)
        return out


def budget_inbox(messages: List[CommsMessage]) -> List[CommsMessage]:
    """
    At most INBOX_MAX_ITEMS for one captain, most useful first.

    LLM-authored traffic outranks the datalink (direct > admiral > fleet
    net > enemy channel, which is capped at INBOX_MAX_ENEMY_ITEMS); datalink
    deaths outrank PD losses outrank launches. Ties keep posting order.
    """
    def rank(item: Tuple[int, CommsMessage]) -> Tuple[int, int, int]:
        index, msg = item
        if msg.sender_role == ROLE_ENGINE:
            return (1, _ENGINE_PRIORITY.get(msg.kind, 3), index)
        return (0, _CHANNEL_PRIORITY.get(msg.channel, 4), index)

    chosen: List[CommsMessage] = []
    enemy = 0
    for _, msg in sorted(enumerate(messages), key=rank):
        if len(chosen) >= INBOX_MAX_ITEMS:
            break
        if msg.channel == CHANNEL_ENEMY:
            if enemy >= INBOX_MAX_ENEMY_ITEMS:
                continue
            enemy += 1
        chosen.append(msg)
    return chosen


# ---------------------------------------------------------------------------
# discuss_with_admiral: the admiral's battle context
# ---------------------------------------------------------------------------

def discussion_context(
    simulation: Any,
    ship_id: Optional[str],
    ship_name: str,
    directive: Optional[str],
    order_text: Optional[str],
    earlier: List[Dict[str, Any]],
    max_enemies: int = 4,
) -> str:
    """
    Compact live picture for an admiral answering one captain's question.

    The reply prompt used to hold only the plan and past orders, so an
    admiral asked "Lancer-3 is 440 km out, not 86 km?" invented that its own
    number was stale. Ranges and closing speeds here are measured from the
    ASKING ship.
    """
    now = getattr(simulation, "current_time", 0.0)
    lines = [f"LIVE BATTLE CONTEXT (T+{now:.0f}s, sensor data - answer from these numbers):"]
    ship = simulation.get_ship(ship_id) if ship_id else None
    if ship is not None:
        ready, total = pd_operational(ship)
        target = simulation.get_ship(ship.primary_target_id) if ship.primary_target_id else None
        maneuver = maneuver_name(ship.current_maneuver.maneuver_type) if ship.current_maneuver else "NONE"
        lines.append(
            f"ASKING SHIP {ship_name} ({ship.ship_type}): hull {ship.hull_integrity:.0f}% | "
            f"heat {ship.heat_percent:.0f}% | PD {ready}/{total} | torpedoes "
            f"{ship.torpedoes_remaining} | speed {ship.velocity.magnitude / 1000:.1f} km/s | "
            f"maneuver {maneuver} | target {getattr(target, 'name', None) or 'none'}")
        enemies = []
        for enemy in simulation.get_enemy_ships(ship_id):
            if not ship_alive(enemy):
                continue
            offset = enemy.position - ship.position
            dist = offset.magnitude
            closing = (-(enemy.velocity - ship.velocity).dot(offset.normalized())
                       if dist > 0 else 0.0)
            enemies.append((dist, closing, enemy))
        enemies.sort(key=lambda e: e[0])
        if enemies:
            lines.append(f"ENEMIES BY RANGE FROM {ship_name}:")
            for dist, closing, enemy in enemies[:max_enemies]:
                trend = (f"closing {closing / 1000:.1f} km/s" if closing > 0
                         else f"separating {-closing / 1000:.1f} km/s")
                lines.append(f"  {getattr(enemy, 'name', enemy.ship_id)} ({enemy.ship_type}): "
                             f"{dist / 1000:.0f} km, {trend}")
        inbound = [eta for eta in (
            torpedo_eta(tf, ship) for tf in getattr(simulation, "torpedoes", None) or []
            if tf.torpedo.target_id == ship_id and not getattr(tf, "is_disabled", False))
            if eta is not None]
        if inbound:
            lines.append(f"LIVE TORPEDOES ON {ship_name}: {len(inbound)}, "
                         f"first impact in ~{min(inbound):.0f}s")
    else:
        lines.append(f"ASKING SHIP {ship_name}: no longer on sensors.")
    lines.append(f"YOUR FLEET DIRECTIVE THIS CHECKPOINT: {clip(directive, 400) if directive else '(none)'}")
    lines.append(f"YOUR ORDER TO {ship_name} THIS CHECKPOINT: {clip(order_text, 400) if order_text else '(none)'}")
    if earlier:
        lines.append(f"EARLIER EXCHANGES WITH {ship_name}:")
        for qa in earlier[-2:]:
            lines.append(f"  T+{qa['t']:.0f}s Q: {clip(qa['question'], 160)} | A: {clip(qa['answer'], 200)}")
    lines.append("Answer in at most 3 short sentences: confirm or amend the order with a concrete "
                 "maneuver/target, and correct the captain only with the numbers above.")
    return "\n".join(lines)
