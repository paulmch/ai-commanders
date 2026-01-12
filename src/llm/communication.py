"""
Captain-to-captain communication system.

Handles message queuing, delivery, and special commands
(surrender, draw proposals).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum, auto


class MessageType(Enum):
    """Types of captain messages."""
    NORMAL = auto()
    SURRENDER = auto()
    PROPOSE_DRAW = auto()
    RETRACT_DRAW = auto()
    ACCEPT_DRAW = auto()


@dataclass
class CaptainMessage:
    """A message between captains."""
    sender_id: str
    sender_name: str
    ship_name: str
    content: str
    timestamp: float
    message_type: MessageType = MessageType.NORMAL

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
        else:
            return f"Captain {self.sender_name}: \"{self.content}\""


@dataclass
class CommunicationChannel:
    """
    Manages communication between two captains.

    Messages are queued during one checkpoint and delivered at the next,
    simulating radio propagation delay.
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
        """
        Queue a message for delivery at next checkpoint.

        Args:
            sender_id: "alpha" or "beta"
            content: Message content
            timestamp: Simulation time
            message_type: Type of message
        """
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
        """
        Get messages for a recipient and mark as delivered.

        Messages are filtered to only include those from the other captain.

        Args:
            recipient_id: "alpha" or "beta"

        Returns:
            List of messages for this recipient
        """
        # Get messages from the OTHER captain
        messages = [
            msg for msg in self.pending_messages
            if msg.sender_id != recipient_id
        ]

        # Move delivered messages
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
