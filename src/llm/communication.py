"""
Communication system for battles with captains and admirals.

Fleet battles: `CommsMessage` + `FleetCommunicationChannel` - every message
has one channel (fleet net, direct, captain<->admiral, enemy/open channel,
admiral diplomacy), faction-checked recipients and a delivery time (captain
traffic arrives at the next checkpoint). The budgets the comms tools
advertise live here as constants.

1v1 duels keep the original `CaptainMessage` / `CommunicationChannel` pair
(messages to the other captain, surrender and draw signalling).
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Dict
from enum import Enum, auto


class MessageType(Enum):
    """Types of messages in the battle."""
    NORMAL = auto()
    SURRENDER = auto()
    PROPOSE_DRAW = auto()
    RETRACT_DRAW = auto()
    ACCEPT_DRAW = auto()
    # Admiral-related types. Fleet traffic now travels as CommsMessage on a
    # channel; ADMIRAL_TO_ADMIRAL still formats diplomacy, and the other two
    # stay so code that classifies legacy CaptainMessages by type (the
    # captain prompt's sender-role mapping) keeps working.
    ADMIRAL_ORDER = auto()
    CAPTAIN_TO_ADMIRAL = auto()
    ADMIRAL_TO_ADMIRAL = auto()
    BROADCAST = auto()  # All captains can see


@dataclass
class CaptainMessage:
    """A message between participants."""
    sender_id: str
    sender_name: str
    ship_name: str  # Ship name or "FLAGSHIP" for admirals
    content: str
    timestamp: float
    message_type: MessageType = MessageType.NORMAL
    recipient_id: Optional[str] = None  # For directed messages

    @property
    def _titled_sender(self) -> str:
        """
        Sender name with exactly one "Captain " prefix.

        Captain names are auto-generated as "Captain <Model>", so blindly
        prepending the title rendered "Captain Captain Kimi" in battle logs and
        recordings.
        """
        name = self.sender_name or ""
        return name if name.startswith("Captain ") else f"Captain {name}"

    def format_for_display(self) -> str:
        """Format message for console output."""
        if self.message_type == MessageType.SURRENDER:
            return f"[SURRENDER] {self._titled_sender} of {self.ship_name} surrenders!"
        elif self.message_type == MessageType.PROPOSE_DRAW:
            return f"[DRAW PROPOSAL] {self._titled_sender} proposes a mutual draw."
        elif self.message_type == MessageType.RETRACT_DRAW:
            return f"[DRAW RETRACTED] {self._titled_sender} retracts their draw proposal."
        elif self.message_type == MessageType.ACCEPT_DRAW:
            return f"[DRAW ACCEPTED] {self._titled_sender} accepts the draw."
        elif self.message_type == MessageType.ADMIRAL_TO_ADMIRAL:
            return f"[ADMIRAL] {self.sender_name}: \"{self.content}\""
        elif self.message_type == MessageType.BROADCAST:
            return f"[BROADCAST] {self.sender_name} ({self.ship_name}): \"{self.content}\""
        else:
            return f"[{self.ship_name}] {self._titled_sender}: \"{self.content}\""

    def format_for_llm(self) -> str:
        """Format message for inclusion in LLM prompt."""
        if self.message_type == MessageType.SURRENDER:
            return f"ENEMY SURRENDERED: {self._titled_sender} has surrendered."
        elif self.message_type == MessageType.PROPOSE_DRAW:
            return f"DRAW PROPOSED: {self._titled_sender} proposes a mutual draw. Use propose_draw tool to accept."
        elif self.message_type == MessageType.RETRACT_DRAW:
            return f"DRAW RETRACTED: {self._titled_sender} has retracted their draw proposal. Battle continues."
        elif self.message_type == MessageType.ADMIRAL_TO_ADMIRAL:
            return f"[ENEMY ADMIRAL] {self.sender_name}: \"{self.content}\""
        elif self.message_type == MessageType.BROADCAST:
            return f"[ALL SHIPS] {self.sender_name}: \"{self.content}\""
        else:
            return f"{self._titled_sender}: \"{self.content}\""


# ---------------------------------------------------------------------------
# Fleet comms (fleet battles)
#
# One message model for every channel a fleet battle has. Routing used to be a
# single "deliver to the OTHER faction unless recipient_id matches" rule, so a
# captain's ALL_FRIENDLIES coordination went to the enemy, a ship NAME never
# matched an id and sat in the queue forever, and nothing could reach an
# admiral. Every message now carries an explicit channel and the resolved,
# faction-checked recipient ids; the runner delivers due messages once per
# checkpoint (all three battle loops share that one delivery function) and
# the recorder writes the comms_message / comms_delivered events from the
# same dicts the captains are handed.
# ---------------------------------------------------------------------------

CHANNEL_FLEET_NET = "fleet_net"                    # friendly broadcast (captain or engine datalink)
CHANNEL_DIRECT = "direct"                          # captain -> one friendly ship
CHANNEL_CAPTAIN_TO_ADMIRAL = "captain_to_admiral"  # captain -> own admiral
CHANNEL_ADMIRAL_TO_CAPTAIN = "admiral_to_captain"  # admiral -> own captain
CHANNEL_ENEMY = "enemy_broadcast"                  # open/enemy channel: taunts, surrender talk
CHANNEL_DIPLOMACY = "admiral_diplomacy"            # admiral <-> admiral (LLM or MCP)

ROLE_CAPTAIN = "captain"
ROLE_ADMIRAL = "admiral"
ROLE_ENGINE = "engine"
ROLE_MCP = "mcp"

# Channels a captain hears from its own side (C renders them as FLEET NET).
FRIENDLY_CHANNELS = (CHANNEL_FLEET_NET, CHANNEL_DIRECT, CHANNEL_ADMIRAL_TO_CAPTAIN)

# --- Budgets. Tool descriptions (tools.py) quote these constants, and
# captain._execute_tool / the inbox builder enforce them, so the advertised
# limits and the real ones cannot drift (tests/test_fleet_comms.py binds them).
MAX_FLEET_SIGNALS_PER_DECISION = 2
MAX_DISCUSSION_EXCHANGES = 2   # discuss_with_admiral questions per checkpoint
MAX_ENEMY_MESSAGES_PER_DECISION = 2
SIGNAL_TEXT_CHARS = 160
REPORT_REASON_CHARS = 200
REPORT_INTENT_CHARS = 120
CONTACT_NOTE_CHARS = 80
REQUEST_TEXT_CHARS = 120
INBOX_MAX_ITEMS = 6
INBOX_TEXT_CHARS = 160
INBOX_MAX_ENEMY_ITEMS = 2

REPORT_STATUSES = ("ACK", "DEVIATE", "UNABLE")
CONTACT_ASSESSMENTS = ("CLOSING", "SEPARATING", "CRIPPLED", "PD_DOWN", "TORPS_INBOUND", "NEW_THREAT")
REQUEST_KINDS = ("PD_COVER", "TORPEDO_SUPPORT", "RETARGET", "FALL_BACK", "REINFORCE")
SIGNAL_KINDS = ("TARGET_CLAIM", "PD_COVER_REQUEST", "TOT_SYNC", "BREAKING_OFF", "FREE")

# Recipient specs understood by FleetCommunicationChannel.resolve().
SPEC_FLEET = "FLEET"
SPEC_ADMIRAL = "ADMIRAL"
SPEC_ENEMIES = "ALL_ENEMIES"

_SPEC_ALIASES = {
    "ALL_FRIENDLIES": SPEC_FLEET, "FRIENDLIES": SPEC_FLEET, "FLEET": SPEC_FLEET,
    "ADMIRAL": SPEC_ADMIRAL,
    "ALL_ENEMIES": SPEC_ENEMIES, "ENEMIES": SPEC_ENEMIES, "ENEMY": SPEC_ENEMIES,
}


def clip(text: Any, limit: int) -> str:
    """One-line, whitespace-normalised text cut to `limit` chars ('...' marks a cut)."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 3)].rstrip() + "..."


def admiral_id(faction: str) -> str:
    """The comms address of a faction's admiral (LLM admiral or MCP commander)."""
    return f"{faction}_admiral"


def _enum_value(value: Any, allowed: tuple, aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Upper-cased `value` if it (or its alias) is one of `allowed`, else None."""
    key = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    key = (aliases or {}).get(key, key)
    return key if key in allowed else None


_STATUS_ALIASES = {"ACKNOWLEDGE": "ACK", "ACKNOWLEDGED": "ACK", "DEVIATION": "DEVIATE"}


def normalize_report(args: Dict[str, Any], legacy: bool = False) -> tuple:
    """
    report_to_admiral (or legacy respond_to_orders) arguments -> (report, errors).

    report: {status, reason, intent, contact, request}, every text clipped to
    its advertised budget. errors: tool-error strings the captain is shown
    next turn (the report is still filed - a DEVIATE without a reason is
    worth more to the Admiral than nothing).
    """
    args = args or {}
    errors: List[str] = []
    if legacy:
        args = {
            "status": args.get("response_type", "ACKNOWLEDGE"),
            "reason": args.get("deviation_reason", ""),
            "intent": args.get("acknowledgment_note", ""),
        }
    status = _enum_value(args.get("status"), REPORT_STATUSES, _STATUS_ALIASES)
    if status is None:
        errors.append(f"report_to_admiral: status must be one of {', '.join(REPORT_STATUSES)} "
                      f"(got {args.get('status')!r}); filed as ACK")
        status = "ACK"
    reason = clip(args.get("reason"), REPORT_REASON_CHARS)
    intent = clip(args.get("intent"), REPORT_INTENT_CHARS)
    if status != "ACK" and not reason:
        errors.append(f"report_to_admiral: {status} needs a reason - the Admiral cannot act "
                      "on a deviation it does not understand")
    contact = None
    raw_contact = args.get("contact")
    if isinstance(raw_contact, dict) and str(raw_contact.get("ship") or "").strip():
        assessment = _enum_value(raw_contact.get("assessment"), CONTACT_ASSESSMENTS)
        if assessment is None:
            errors.append("report_to_admiral: contact.assessment must be one of "
                          + ", ".join(CONTACT_ASSESSMENTS))
        contact = {"ship": clip(raw_contact.get("ship"), 60),
                   "assessment": assessment or "NEW_THREAT",
                   "note": clip(raw_contact.get("note"), CONTACT_NOTE_CHARS)}
    request = None
    raw_request = args.get("request")
    if isinstance(raw_request, dict) and raw_request.get("kind"):
        kind = _enum_value(raw_request.get("kind"), REQUEST_KINDS)
        if kind is None:
            errors.append("report_to_admiral: request.kind must be one of " + ", ".join(REQUEST_KINDS))
        else:
            request = {"kind": kind, "text": clip(raw_request.get("text"), REQUEST_TEXT_CHARS)}
    report = {"status": status, "reason": reason, "intent": intent,
              "contact": contact, "request": request}
    return report, errors


def normalize_signal(args: Dict[str, Any]) -> tuple:
    """fleet_signal arguments -> (signal or None, errors)."""
    args = args or {}
    errors: List[str] = []
    kind = _enum_value(args.get("kind"), SIGNAL_KINDS) or "FREE"
    text = clip(args.get("text") or args.get("message"), SIGNAL_TEXT_CHARS)
    if not text:
        return None, ["fleet_signal: text is required - nothing was sent"]
    eta_s = args.get("eta_s")
    try:
        eta_s = None if eta_s in (None, "") else max(0.0, float(eta_s))
    except (TypeError, ValueError):
        errors.append(f"fleet_signal: eta_s must be a number of seconds (got {eta_s!r})")
        eta_s = None
    signal = {"kind": kind, "to": str(args.get("to") or "ALL").strip() or "ALL",
              "target": clip(args.get("target"), 60) or None, "eta_s": eta_s, "text": text}
    return signal, errors


@dataclass
class CommsMessage:
    """
    One fleet-battle message, on exactly one channel.

    `to_dict()` is the recorded `comms_message` payload and, plus
    `delivered_t`, the item a captain finds in `captain.inbox`.
    """
    msg_id: str
    channel: str
    kind: str
    sender_id: str
    sender_role: str
    sender_name: str
    faction: str
    recipient_spec: str
    recipient_ids: List[str]
    text: str
    structured: Dict[str, Any] = field(default_factory=dict)
    checkpoint: int = 0
    sent_t: float = 0.0
    deliver_at_t: float = 0.0
    # Delivery bookkeeping (not part of the recorded payload).
    delivered_to: List[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        """Legacy CaptainMessage name for the text."""
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "channel": self.channel,
            "kind": self.kind,
            "sender_id": self.sender_id,
            "sender_role": self.sender_role,
            "sender_name": self.sender_name,
            "faction": self.faction,
            "recipient_spec": self.recipient_spec,
            "recipient_ids": list(self.recipient_ids),
            "text": self.text,
            "structured": dict(self.structured),
            "checkpoint": self.checkpoint,
            "sent_t": self.sent_t,
            "deliver_at_t": self.deliver_at_t,
        }

    def inbox_item(self, delivered_t: float) -> Dict[str, Any]:
        """What a recipient captain is handed: the payload plus delivery time."""
        item = self.to_dict()
        item["text"] = clip(item["text"], INBOX_TEXT_CHARS)
        item["delivered_t"] = delivered_t
        return item

    def format_for_display(self) -> str:
        """Console / BattleResult line."""
        to = self.recipient_spec or ",".join(self.recipient_ids)
        return f"[{self.channel}] {self.sender_name} -> {to}: \"{self.text}\""


@dataclass
class Resolution:
    """Outcome of resolving a recipient spec: a channel and recipient ids, or an error."""
    channel: str = ""
    recipient_ids: List[str] = field(default_factory=list)
    recipient_spec: str = ""
    error: Optional[str] = None


@dataclass
class FleetCommunicationChannel:
    """
    Fleet-battle message routing and delivery.

    Ships and admirals are registered at setup. A message is posted with its
    channel and resolved recipient ids and becomes due at `deliver_at_t`
    (captain traffic: the next checkpoint). `deliver_due()` hands each due
    message to every recipient that is still in the fight exactly once, then
    drops it - recipients that died in the meantime are skipped, nothing is
    left in the queue to leak into later checkpoints.
    """

    # Ship registry: ship_id -> (captain_name, ship_name, faction)
    ships: Dict[str, tuple] = field(default_factory=dict)

    # Admiral registry: faction -> admiral/MCP commander name
    admirals: Dict[str, str] = field(default_factory=dict)

    pending_messages: List[CommsMessage] = field(default_factory=list)
    all_messages: List[CommsMessage] = field(default_factory=list)

    # Per-sender message counter for deterministic, readable msg ids. Locked:
    # discussion traffic is recorded from the parallel captain threads.
    _counters: Dict[str, int] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def register_ship(
        self,
        ship_id: str,
        captain_name: str,
        ship_name: str,
        faction: str,
    ) -> None:
        """Register a ship with the communication system."""
        self.ships[ship_id] = (captain_name, ship_name, faction)

    def register_admiral(self, faction: str, admiral_name: str) -> None:
        """Register an admiral (LLM admiral or MCP commander) for a faction."""
        self.admirals[faction] = admiral_name

    def get_faction(self, ship_id: str) -> Optional[str]:
        """Faction of a registered ship or admiral id."""
        if ship_id in self.ships:
            return self.ships[ship_id][2]
        if ship_id.endswith("_admiral") and ship_id[:-len("_admiral")] in self.admirals:
            return ship_id[:-len("_admiral")]
        return None

    def ship_name(self, ship_id: str) -> str:
        """Display name of a registered ship (the id when unknown)."""
        entry = self.ships.get(ship_id)
        return entry[1] if entry else ship_id

    def sender_name(self, sender_id: str) -> str:
        """Display name for a sender id (ship name, admiral name or the id)."""
        if sender_id in self.ships:
            return self.ships[sender_id][1]
        faction = self.get_faction(sender_id)
        if faction and sender_id == admiral_id(faction):
            return self.admirals.get(faction, f"Admiral {faction.title()}")
        return sender_id

    def faction_ship_ids(self, faction: str) -> List[str]:
        return [sid for sid, (_, _, f) in self.ships.items() if f == faction]

    # ------------------------------------------------------------------
    # Recipient resolution
    # ------------------------------------------------------------------

    def match_ship(
        self,
        name: str,
        prefer: Optional[Callable[[str], bool]] = None,
        exclude: Optional[str] = None,
    ) -> List[str]:
        """
        Ship ids a free-text ship reference can mean.

        Matches (case-insensitive) the ship id, the full name, or the name
        with its hull prefix stripped ("TIS Dart-1" -> "Dart-1"). When several
        ships match (battle B had a friendly TIS Lancer-1 and an enemy OCS
        Lancer-1) the ones satisfying `prefer` win; a full name with its
        prefix, or the id, is always unambiguous.
        """
        key = " ".join(str(name or "").split()).lower()
        if not key:
            return []
        # The label form the turn states teach, "OCS Lancer-1 [beta_2]": the id wins.
        if key.endswith("]") and "[" in key:
            label_id = key[key.rindex("[") + 1:-1].strip()
            ids = [sid for sid in self.ships if sid.lower() == label_id and sid != exclude]
            if ids:
                return ids
            key = key[:key.rindex("[")].strip()
        exact, short = [], []
        for sid, (_, ship_name, _) in self.ships.items():
            if sid == exclude:
                continue
            full = " ".join(str(ship_name or "").split()).lower()
            if key == sid.lower() or key == full:
                exact.append(sid)
            elif " " in full and key == full.split(" ", 1)[1]:
                short.append(sid)
        found = exact or short
        if len(found) > 1 and prefer:
            preferred = [sid for sid in found if prefer(sid)]
            if preferred:
                found = preferred
        return found

    def resolve(
        self,
        sender_id: str,
        spec: Optional[str],
        *,
        friendly_only: bool,
        alive: Optional[Callable[[str], bool]] = None,
    ) -> Resolution:
        """
        Turn a recipient spec into a channel and faction-checked recipient ids.

        spec: FLEET / ALL_FRIENDLIES (every living friendly ship but the
        sender), ADMIRAL (own admiral), ALL_ENEMIES (every living enemy ship),
        or a ship id / name. friendly_only (fleet_signal) refuses enemy
        targets instead of transmitting to them; send_message
        (friendly_only=False) addressed to a FRIENDLY ship still stays on the
        friendly side (direct), never on the enemy channel.
        """
        faction = self.get_faction(sender_id)
        if faction is None:
            return Resolution(error=f"unknown sender {sender_id!r}")
        alive = alive or (lambda _sid: True)
        raw = " ".join(str(spec or "").split())
        upper = raw.upper().replace(" ", "_")
        if upper in ("", "ALL"):
            # fleet_signal's ALL is the fleet net; send_message's is the open
            # (enemy) channel - its old ALL already only ever reached enemies.
            canonical = SPEC_FLEET if friendly_only else SPEC_ENEMIES
        else:
            canonical = _SPEC_ALIASES.get(upper)

        if canonical == SPEC_FLEET:
            ids = [sid for sid in self.faction_ship_ids(faction) if sid != sender_id and alive(sid)]
            if not ids:
                return Resolution(error="no other friendly ship is left to hear it")
            return Resolution(CHANNEL_FLEET_NET, ids, "ALL")
        if canonical == SPEC_ADMIRAL:
            if faction not in self.admirals:
                return Resolution(error="your fleet has no Admiral")
            return Resolution(CHANNEL_CAPTAIN_TO_ADMIRAL, [admiral_id(faction)], "ADMIRAL")
        if canonical == SPEC_ENEMIES:
            if friendly_only:
                return Resolution(error="fleet_signal reaches only your own fleet; "
                                        "use send_message for the enemy channel")
            ids = [sid for sid, (_, _, f) in self.ships.items() if f != faction and alive(sid)]
            if not ids:
                return Resolution(error="no enemy ship is left to hear it")
            return Resolution(CHANNEL_ENEMY, ids, "ALL_ENEMIES")

        # fleet_signal means a friendly ship, send_message an enemy one.
        matches = self.match_ship(
            raw, exclude=sender_id,
            prefer=lambda sid: (self.ships[sid][2] == faction) == friendly_only)
        if not matches and self.match_ship(raw):
            return Resolution(error="that is your own ship")
        if not matches:
            valid = ", ".join(self.ships[sid][1] for sid in self.faction_ship_ids(faction)
                              if sid != sender_id) or "none"
            return Resolution(error=f"no ship named {raw!r} (your fleet: {valid}; "
                                    "or use ALL / ADMIRAL)")
        if len(matches) > 1:
            names = ", ".join(f"{self.ships[s][1]} ({s})" for s in matches)
            return Resolution(error=f"{raw!r} is ambiguous: {names} - use the ship id")
        target = matches[0]
        target_name = self.ships[target][1]
        if not alive(target):
            return Resolution(error=f"{target_name} is no longer in the fight")
        if self.ships[target][2] == faction:
            return Resolution(CHANNEL_DIRECT, [target], target_name)
        if friendly_only:
            return Resolution(error=f"{target_name} is an ENEMY ship; fleet_signal reaches only "
                                    "your own fleet (send_message is the enemy channel)")
        return Resolution(CHANNEL_ENEMY, [target], target_name)

    # ------------------------------------------------------------------
    # Posting and delivery
    # ------------------------------------------------------------------

    def next_msg_id(self, checkpoint: int, sender_id: str) -> str:
        with self._lock:
            n = self._counters.get(sender_id, 0) + 1
            self._counters[sender_id] = n
        return f"m{checkpoint}-{sender_id}-{n}"

    def post(self, message: CommsMessage) -> CommsMessage:
        """Queue a message for delivery at its deliver_at_t."""
        if not message.msg_id:
            message.msg_id = self.next_msg_id(message.checkpoint, message.sender_id)
        with self._lock:
            self.pending_messages.append(message)
            self.all_messages.append(message)
        return message

    def log(self, message: CommsMessage) -> CommsMessage:
        """Keep a message delivered outside the queue (discussion, diplomacy, reports)."""
        if not message.msg_id:
            message.msg_id = self.next_msg_id(message.checkpoint, message.sender_id)
        with self._lock:
            self.all_messages.append(message)
        return message

    def due_messages(self, now: float) -> List[CommsMessage]:
        """Messages whose delivery time has come, in posting order."""
        return [m for m in self.pending_messages if m.deliver_at_t <= now + 1e-6]

    def deliver_due(
        self,
        now: float,
        recipients: List[str],
    ) -> Dict[str, List[CommsMessage]]:
        """
        Hand every due message to each of its recipients in `recipients`.

        `recipients` is the set of ids able to read this checkpoint (living,
        not surrendered, LLM-commanded ships). Due messages are then dropped:
        a recipient that is gone by now never gets them, and nothing lingers.
        """
        out: Dict[str, List[CommsMessage]] = {rid: [] for rid in recipients}
        due = self.due_messages(now)
        for msg in due:
            for rid in msg.recipient_ids:
                if rid in out and rid not in msg.delivered_to:
                    out[rid].append(msg)
                    msg.delivered_to.append(rid)
        if due:
            due_ids = {id(m) for m in due}
            self.pending_messages = [m for m in self.pending_messages if id(m) not in due_ids]
        return out

    # ------------------------------------------------------------------
    # Legacy API (pre-channel callers and tests)
    # ------------------------------------------------------------------

    def queue_message(
        self,
        sender_id: str,
        content: str,
        timestamp: float,
        message_type: MessageType = MessageType.NORMAL,
        recipient_id: Optional[str] = None,
    ) -> CommsMessage:
        """
        Queue a message the old way: no recipient = the enemy channel.

        A recipient id is now faction-checked: a friendly id goes out on the
        direct channel instead of being lumped in with enemy traffic.
        """
        faction = self.get_faction(sender_id) or ""
        if recipient_id:
            friendly = self.get_faction(recipient_id) == faction
            channel = CHANNEL_DIRECT if friendly else CHANNEL_ENEMY
            ids, spec = [recipient_id], self.ship_name(recipient_id)
        else:
            channel = CHANNEL_ENEMY
            ids = [sid for sid, (_, _, f) in self.ships.items() if f != faction]
            spec = "ALL_ENEMIES"
        return self.post(CommsMessage(
            msg_id="", channel=channel, kind=message_type.name,
            sender_id=sender_id, sender_role=ROLE_CAPTAIN,
            sender_name=self.sender_name(sender_id), faction=faction,
            recipient_spec=spec, recipient_ids=ids, text=content,
            sent_t=timestamp, deliver_at_t=timestamp,
        ))

    def deliver_messages(self, recipient_id: str) -> List[CommsMessage]:
        """Every pending message for one recipient, regardless of due time."""
        mine = [m for m in self.pending_messages
                if recipient_id in m.recipient_ids and recipient_id not in m.delivered_to]
        for msg in mine:
            msg.delivered_to.append(recipient_id)
        self.pending_messages = [
            m for m in self.pending_messages
            if any(r not in m.delivered_to for r in m.recipient_ids)
        ]
        return mine

    def get_all_messages_formatted(self) -> List[str]:
        """Every authored message (datalink telemetry left out), for BattleResult."""
        return [msg.format_for_display() for msg in self.all_messages
                if msg.sender_role != ROLE_ENGINE]


# Legacy compatibility - simple two-captain channel
@dataclass
class CommunicationChannel:
    """
    Simple communication between two captains.

    Kept for backward compatibility. Use FleetCommunicationChannel
    for multi-ship battles.
    """
    alpha_name: str
    alpha_ship: str
    beta_name: str
    beta_ship: str

    # Message queues
    pending_messages: List[CaptainMessage] = field(default_factory=list)
    delivered_messages: List[CaptainMessage] = field(default_factory=list)
    all_messages: List[CaptainMessage] = field(default_factory=list)

    # State tracking
    alpha_surrendered: bool = False
    beta_surrendered: bool = False
    alpha_proposed_draw: bool = False
    beta_proposed_draw: bool = False

    def queue_message(
        self,
        sender_id: str,
        content: str,
        timestamp: float,
        message_type: MessageType = MessageType.NORMAL,
    ) -> None:
        """Queue a message for delivery at next checkpoint."""
        if sender_id == "alpha":
            sender_name = self.alpha_name
            ship_name = self.alpha_ship
        else:
            sender_name = self.beta_name
            ship_name = self.beta_ship

        message = CaptainMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            ship_name=ship_name,
            content=content,
            timestamp=timestamp,
            message_type=message_type,
        )

        self.pending_messages.append(message)
        self.all_messages.append(message)

        # Track special message types
        if message_type == MessageType.SURRENDER:
            if sender_id == "alpha":
                self.alpha_surrendered = True
            else:
                self.beta_surrendered = True
        elif message_type == MessageType.PROPOSE_DRAW:
            if sender_id == "alpha":
                self.alpha_proposed_draw = True
            else:
                self.beta_proposed_draw = True
        elif message_type == MessageType.RETRACT_DRAW:
            if sender_id == "alpha":
                self.alpha_proposed_draw = False
            else:
                self.beta_proposed_draw = False

    def deliver_messages(self, recipient_id: str) -> List[CaptainMessage]:
        """Get messages for a recipient and mark as delivered."""
        messages = [
            msg for msg in self.pending_messages
            if msg.sender_id != recipient_id
        ]

        for msg in messages:
            self.delivered_messages.append(msg)
            self.pending_messages.remove(msg)

        return messages

    def format_messages_for_llm(self, messages: List[CaptainMessage]) -> str:
        """Format messages for inclusion in LLM prompt."""
        if not messages:
            return ""
        return "\n".join(msg.format_for_llm() for msg in messages)

    def is_battle_ended(self) -> bool:
        """Check if battle should end due to communication."""
        return self.has_surrender() or self.has_mutual_draw()

    def has_surrender(self) -> bool:
        """Check if either captain surrendered."""
        return self.alpha_surrendered or self.beta_surrendered

    def has_mutual_draw(self) -> bool:
        """Check if both captains agreed to draw."""
        return self.alpha_proposed_draw and self.beta_proposed_draw

    def get_surrender_loser(self) -> Optional[str]:
        """Get the ID of the captain who surrendered."""
        if self.alpha_surrendered:
            return "alpha"
        elif self.beta_surrendered:
            return "beta"
        return None

    def get_all_messages_formatted(self) -> List[str]:
        """Get all messages formatted for display."""
        return [msg.format_for_display() for msg in self.all_messages]
