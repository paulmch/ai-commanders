"""
Communication system for battles with captains and admirals.

Handles message queuing, delivery, and special commands
(surrender, draw proposals). Supports both delayed (30s)
and immediate messaging.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum, auto


class MessageType(Enum):
    """Types of messages in the battle."""
    NORMAL = auto()
    SURRENDER = auto()
    PROPOSE_DRAW = auto()
    RETRACT_DRAW = auto()
    ACCEPT_DRAW = auto()
    # Admiral-related message types
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

    def format_for_display(self) -> str:
        """Format message for console output."""
        if self.message_type == MessageType.SURRENDER:
            return f"[SURRENDER] Captain {self.sender_name} of {self.ship_name} surrenders!"
        elif self.message_type == MessageType.PROPOSE_DRAW:
            return f"[DRAW PROPOSAL] Captain {self.sender_name} proposes a mutual draw."
        elif self.message_type == MessageType.RETRACT_DRAW:
            return f"[DRAW RETRACTED] Captain {self.sender_name} retracts their draw proposal."
        elif self.message_type == MessageType.ACCEPT_DRAW:
            return f"[DRAW ACCEPTED] Captain {self.sender_name} accepts the draw."
        elif self.message_type == MessageType.ADMIRAL_TO_ADMIRAL:
            return f"[ADMIRAL] {self.sender_name}: \"{self.content}\""
        elif self.message_type == MessageType.BROADCAST:
            return f"[BROADCAST] {self.sender_name} ({self.ship_name}): \"{self.content}\""
        else:
            return f"[{self.ship_name}] Captain {self.sender_name}: \"{self.content}\""

    def format_for_llm(self) -> str:
        """Format message for inclusion in LLM prompt."""
        if self.message_type == MessageType.SURRENDER:
            return f"ENEMY SURRENDERED: Captain {self.sender_name} has surrendered."
        elif self.message_type == MessageType.PROPOSE_DRAW:
            return f"DRAW PROPOSED: Captain {self.sender_name} proposes a mutual draw. Use propose_draw tool to accept."
        elif self.message_type == MessageType.RETRACT_DRAW:
            return f"DRAW RETRACTED: Captain {self.sender_name} has retracted their draw proposal. Battle continues."
        elif self.message_type == MessageType.ADMIRAL_TO_ADMIRAL:
            return f"[ENEMY ADMIRAL] {self.sender_name}: \"{self.content}\""
        elif self.message_type == MessageType.BROADCAST:
            return f"[ALL SHIPS] {self.sender_name}: \"{self.content}\""
        else:
            return f"Captain {self.sender_name}: \"{self.content}\""


@dataclass
class FleetCommunicationChannel:
    """
    Enhanced communication system supporting multiple ships and admirals.

    Supports:
    - Delayed messages (30s, delivered at next checkpoint)
    - Immediate messages (within checkpoint: Admiral-Admiral, Captain-Admiral, broadcast)
    - Multi-ship messaging with faction awareness
    """

    # Ship registry: ship_id -> (captain_name, ship_name, faction)
    ships: Dict[str, tuple] = field(default_factory=dict)

    # Admiral registry: faction -> (admiral_name,)
    admirals: Dict[str, str] = field(default_factory=dict)

    # Message queues
    pending_messages: List[CaptainMessage] = field(default_factory=list)  # 30s delay
    delivered_messages: List[CaptainMessage] = field(default_factory=list)
    all_messages: List[CaptainMessage] = field(default_factory=list)

    # Immediate message queue (processed within checkpoint)
    immediate_queue: List[CaptainMessage] = field(default_factory=list)

    # Broadcast messages (visible to all)
    broadcast_messages: List[CaptainMessage] = field(default_factory=list)

    # State tracking per ship
    surrendered_ships: set = field(default_factory=set)

    # Captain draw proposals (only when no Admiral)
    captain_proposed_draw: Dict[str, bool] = field(default_factory=dict)

    # Admiral draw proposals
    alpha_admiral_proposed_draw: bool = False
    beta_admiral_proposed_draw: bool = False

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
        """Register an admiral."""
        self.admirals[faction] = admiral_name

    def get_faction(self, ship_id: str) -> Optional[str]:
        """Get the faction of a ship."""
        if ship_id in self.ships:
            return self.ships[ship_id][2]
        return None

    def queue_message(
        self,
        sender_id: str,
        content: str,
        timestamp: float,
        message_type: MessageType = MessageType.NORMAL,
        recipient_id: Optional[str] = None,
    ) -> None:
        """
        Queue a message for delivery at next checkpoint (30s delay).

        Args:
            sender_id: Ship ID or "alpha_admiral"/"beta_admiral"
            content: Message content
            timestamp: Simulation time
            message_type: Type of message
            recipient_id: Optional specific recipient
        """
        sender_name, ship_name = self._get_sender_info(sender_id)

        message = CaptainMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            ship_name=ship_name,
            content=content,
            timestamp=timestamp,
            message_type=message_type,
            recipient_id=recipient_id,
        )

        self.pending_messages.append(message)
        self.all_messages.append(message)

        # Track special message types
        self._handle_special_message(sender_id, message_type)

    def queue_immediate_message(
        self,
        sender_id: str,
        content: str,
        timestamp: float,
        message_type: MessageType = MessageType.NORMAL,
        recipient_id: Optional[str] = None,
    ) -> CaptainMessage:
        """
        Queue a message for immediate delivery within this checkpoint.

        Used for:
        - Admiral <-> Admiral negotiation
        - Captain <-> Admiral discussion
        - Captain <-> Enemy Captain chat (broadcast)

        Args:
            sender_id: Ship ID or "alpha_admiral"/"beta_admiral"
            content: Message content
            timestamp: Simulation time
            message_type: Type of message
            recipient_id: Optional specific recipient

        Returns:
            The created message
        """
        sender_name, ship_name = self._get_sender_info(sender_id)

        message = CaptainMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            ship_name=ship_name,
            content=content,
            timestamp=timestamp,
            message_type=message_type,
            recipient_id=recipient_id,
        )

        self.immediate_queue.append(message)
        self.all_messages.append(message)

        # Track special message types
        self._handle_special_message(sender_id, message_type)

        return message

    def broadcast(
        self,
        sender_id: str,
        content: str,
        timestamp: float,
    ) -> CaptainMessage:
        """
        Send a broadcast message to all ships.

        All captains (friend and foe) can see and respond.

        Args:
            sender_id: Ship ID sending the broadcast
            content: Message content
            timestamp: Simulation time

        Returns:
            The created message
        """
        sender_name, ship_name = self._get_sender_info(sender_id)

        message = CaptainMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            ship_name=ship_name,
            content=content,
            timestamp=timestamp,
            message_type=MessageType.BROADCAST,
        )

        self.broadcast_messages.append(message)
        self.all_messages.append(message)

        return message

    def deliver_messages(self, recipient_id: str) -> List[CaptainMessage]:
        """
        Get delayed messages for a recipient (from previous checkpoint).

        Args:
            recipient_id: Ship ID to receive messages

        Returns:
            List of messages for this recipient
        """
        recipient_faction = self.get_faction(recipient_id)
        if not recipient_faction:
            return []

        # Get messages from OTHER faction
        messages = [
            msg for msg in self.pending_messages
            if self._is_message_for(msg, recipient_id, recipient_faction)
        ]

        # Move delivered messages
        for msg in messages:
            self.delivered_messages.append(msg)
            self.pending_messages.remove(msg)

        return messages

    def deliver_immediate_messages(
        self,
        recipient_id: str,
    ) -> List[CaptainMessage]:
        """
        Get immediate messages for a recipient.

        Args:
            recipient_id: Ship ID or Admiral ID

        Returns:
            List of immediate messages for this recipient
        """
        recipient_faction = self._get_faction_for_id(recipient_id)

        messages = [
            msg for msg in self.immediate_queue
            if self._is_immediate_message_for(msg, recipient_id, recipient_faction)
        ]

        # Remove delivered messages
        for msg in messages:
            self.immediate_queue.remove(msg)

        return messages

    def get_broadcast_messages(
        self,
        since_last_clear: bool = True,
    ) -> List[CaptainMessage]:
        """
        Get broadcast messages.

        Args:
            since_last_clear: If True, clears after returning

        Returns:
            List of broadcast messages
        """
        messages = self.broadcast_messages.copy()
        if since_last_clear:
            self.broadcast_messages.clear()
        return messages

    def clear_immediate_queue(self) -> None:
        """Clear immediate message queue at end of checkpoint."""
        self.immediate_queue.clear()

    def clear_broadcast_messages(self) -> None:
        """Clear broadcast messages at end of checkpoint."""
        self.broadcast_messages.clear()

    def mark_ship_surrendered(self, ship_id: str) -> None:
        """Mark a ship as surrendered."""
        self.surrendered_ships.add(ship_id)

    def is_ship_surrendered(self, ship_id: str) -> bool:
        """Check if a ship has surrendered."""
        return ship_id in self.surrendered_ships

    def set_admiral_draw_proposal(self, faction: str, proposed: bool) -> None:
        """Set Admiral's draw proposal state."""
        if faction == "alpha":
            self.alpha_admiral_proposed_draw = proposed
        else:
            self.beta_admiral_proposed_draw = proposed

    def has_admiral_mutual_draw(self) -> bool:
        """Check if both Admirals agreed to draw."""
        return self.alpha_admiral_proposed_draw and self.beta_admiral_proposed_draw

    def is_battle_ended(self) -> bool:
        """Check if battle should end due to communication."""
        return self.has_any_surrender() or self.has_mutual_draw()

    def has_any_surrender(self) -> bool:
        """Check if any ship surrendered."""
        return len(self.surrendered_ships) > 0

    def has_mutual_draw(self) -> bool:
        """
        Check if both sides agreed to draw.

        Uses Admiral draw if Admirals exist, otherwise captain draws.
        """
        # If either side has Admiral, use Admiral draws
        if self.admirals:
            return self.has_admiral_mutual_draw()

        # Otherwise check captain draws
        alpha_proposed = any(
            self.captain_proposed_draw.get(ship_id, False)
            for ship_id, (_, _, faction) in self.ships.items()
            if faction == "alpha"
        )
        beta_proposed = any(
            self.captain_proposed_draw.get(ship_id, False)
            for ship_id, (_, _, faction) in self.ships.items()
            if faction == "beta"
        )
        return alpha_proposed and beta_proposed

    def get_all_messages_formatted(self) -> List[str]:
        """Get all messages formatted for display."""
        return [msg.format_for_display() for msg in self.all_messages]

    def _get_sender_info(self, sender_id: str) -> tuple:
        """Get sender name and ship name for a sender ID."""
        if sender_id.endswith("_admiral"):
            faction = sender_id.replace("_admiral", "")
            admiral_name = self.admirals.get(faction, f"Admiral {faction.title()}")
            return admiral_name, "FLAGSHIP"

        if sender_id in self.ships:
            captain_name, ship_name, _ = self.ships[sender_id]
            return captain_name, ship_name

        return sender_id, sender_id

    def _get_faction_for_id(self, id_: str) -> Optional[str]:
        """Get faction for any ID (ship or admiral)."""
        if id_.endswith("_admiral"):
            return id_.replace("_admiral", "")
        return self.get_faction(id_)

    def _is_message_for(
        self,
        msg: CaptainMessage,
        recipient_id: str,
        recipient_faction: str,
    ) -> bool:
        """Check if a message is intended for a recipient."""
        # If specific recipient, check match
        if msg.recipient_id:
            return msg.recipient_id == recipient_id

        # Otherwise, deliver if from different faction
        sender_faction = self._get_faction_for_id(msg.sender_id)
        return sender_faction != recipient_faction

    def _is_immediate_message_for(
        self,
        msg: CaptainMessage,
        recipient_id: str,
        recipient_faction: Optional[str],
    ) -> bool:
        """Check if an immediate message is for a recipient."""
        # Directed messages
        if msg.recipient_id:
            return msg.recipient_id == recipient_id

        # Admiral-to-Admiral: only for opposite Admiral
        if msg.message_type == MessageType.ADMIRAL_TO_ADMIRAL:
            sender_faction = self._get_faction_for_id(msg.sender_id)
            return (
                recipient_id.endswith("_admiral") and
                recipient_faction != sender_faction
            )

        # Other immediate messages: deliver to opposite faction
        sender_faction = self._get_faction_for_id(msg.sender_id)
        return sender_faction != recipient_faction

    def _handle_special_message(
        self,
        sender_id: str,
        message_type: MessageType,
    ) -> None:
        """Handle special message type side effects."""
        if message_type == MessageType.SURRENDER:
            self.surrendered_ships.add(sender_id)

        elif message_type == MessageType.PROPOSE_DRAW:
            if sender_id.endswith("_admiral"):
                faction = sender_id.replace("_admiral", "")
                self.set_admiral_draw_proposal(faction, True)
            else:
                self.captain_proposed_draw[sender_id] = True

        elif message_type == MessageType.RETRACT_DRAW:
            if sender_id.endswith("_admiral"):
                faction = sender_id.replace("_admiral", "")
                self.set_admiral_draw_proposal(faction, False)
            else:
                self.captain_proposed_draw[sender_id] = False


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
