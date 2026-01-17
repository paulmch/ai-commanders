"""
MCP Admiral Chat System - Manages inter-admiral communication.

Features:
- 3 messages per side per 30-second turn (checkpoint)
- History kept for last 5 turns
- Thread-safe for MCP integration
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque


@dataclass
class ChatMessage:
    """A single chat message between admirals."""
    turn: int  # Checkpoint number when sent
    timestamp: float  # Simulation time
    sender_faction: str  # "alpha" or "beta"
    content: str
    recipient: str = "enemy"  # "enemy" or specific identifier

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "turn": self.turn,
            "timestamp": self.timestamp,
            "sender": "friendly" if self.sender_faction == "self" else "enemy",
            "sender_faction": self.sender_faction,
            "content": self.content,
            "recipient": self.recipient,
        }


class AdmiralChat:
    """
    Manages chat communication between admirals (MCP or LLM).

    Enforces limits:
    - max_messages_per_side_per_turn: 3 (each side can send 3 messages per checkpoint)
    - history_turns: 5 (keep messages from last 5 turns visible)
    """

    def __init__(
        self,
        max_messages_per_side_per_turn: int = 3,
        history_turns: int = 5,
    ):
        """
        Initialize chat system.

        Args:
            max_messages_per_side_per_turn: Max messages each side can send per turn
            history_turns: Number of turns of history to keep visible
        """
        self.max_messages_per_side_per_turn = max_messages_per_side_per_turn
        self.history_turns = history_turns

        # Thread safety
        self._lock = threading.Lock()

        # Current turn tracking
        self._current_turn: int = 0

        # Messages sent this turn per faction
        self._messages_this_turn: Dict[str, int] = {
            "alpha": 0,
            "beta": 0,
        }

        # Full message history (all messages)
        self._all_messages: List[ChatMessage] = []

        # Pending messages to deliver (per faction)
        self._pending_delivery: Dict[str, List[ChatMessage]] = {
            "alpha": [],
            "beta": [],
        }

    def can_send(self, faction: str) -> bool:
        """
        Check if faction can send another message this turn.

        Args:
            faction: "alpha" or "beta"

        Returns:
            True if faction hasn't exceeded message limit
        """
        with self._lock:
            return self._messages_this_turn.get(faction, 0) < self.max_messages_per_side_per_turn

    def messages_remaining(self, faction: str) -> int:
        """
        Get number of messages faction can still send this turn.

        Args:
            faction: "alpha" or "beta"

        Returns:
            Number of messages remaining
        """
        with self._lock:
            sent = self._messages_this_turn.get(faction, 0)
            return max(0, self.max_messages_per_side_per_turn - sent)

    def send_message(
        self,
        faction: str,
        content: str,
        timestamp: float,
        recipient: str = "enemy",
    ) -> bool:
        """
        Send a message from a faction.

        Args:
            faction: Sending faction ("alpha" or "beta")
            content: Message content
            timestamp: Current simulation time
            recipient: Message recipient (default "enemy")

        Returns:
            True if message was sent, False if limit exceeded
        """
        with self._lock:
            # Check limit
            if self._messages_this_turn.get(faction, 0) >= self.max_messages_per_side_per_turn:
                return False

            # Create message
            message = ChatMessage(
                turn=self._current_turn,
                timestamp=timestamp,
                sender_faction=faction,
                content=content,
                recipient=recipient,
            )

            # Add to history
            self._all_messages.append(message)

            # Queue for delivery to enemy
            enemy_faction = "beta" if faction == "alpha" else "alpha"
            self._pending_delivery[enemy_faction].append(message)

            # Increment counter
            self._messages_this_turn[faction] = self._messages_this_turn.get(faction, 0) + 1

            return True

    def get_pending_messages(self, faction: str) -> List[ChatMessage]:
        """
        Get and clear pending messages for a faction.

        Args:
            faction: Receiving faction

        Returns:
            List of pending messages
        """
        with self._lock:
            messages = self._pending_delivery.get(faction, [])
            self._pending_delivery[faction] = []
            return messages

    def peek_pending_messages(self, faction: str) -> List[ChatMessage]:
        """
        Get pending messages without clearing.

        Args:
            faction: Receiving faction

        Returns:
            List of pending messages
        """
        with self._lock:
            return list(self._pending_delivery.get(faction, []))

    def get_recent_history(self, faction: str) -> List[Dict]:
        """
        Get recent chat history from a faction's perspective.

        Messages are labeled as "friendly" or "enemy" relative to the viewer.

        Args:
            faction: Viewing faction

        Returns:
            List of message dictionaries with perspective-relative sender
        """
        with self._lock:
            # Filter to recent turns
            min_turn = max(0, self._current_turn - self.history_turns)
            recent = [m for m in self._all_messages if m.turn >= min_turn]

            # Convert with perspective
            result = []
            for msg in recent:
                msg_dict = {
                    "turn": msg.turn,
                    "timestamp": msg.timestamp,
                    "sender": "friendly" if msg.sender_faction == faction else "enemy",
                    "content": msg.content,
                }
                result.append(msg_dict)

            return result

    def get_all_history(self) -> List[ChatMessage]:
        """
        Get complete chat history.

        Returns:
            List of all messages
        """
        with self._lock:
            return list(self._all_messages)

    def new_turn(self) -> None:
        """
        Advance to new turn, resetting per-turn message counts.

        Called at the start of each checkpoint.
        """
        with self._lock:
            self._current_turn += 1
            self._messages_this_turn = {
                "alpha": 0,
                "beta": 0,
            }

    def get_current_turn(self) -> int:
        """Get current turn number."""
        with self._lock:
            return self._current_turn

    def reset(self) -> None:
        """Reset chat state (for new battle)."""
        with self._lock:
            self._current_turn = 0
            self._messages_this_turn = {"alpha": 0, "beta": 0}
            self._all_messages = []
            self._pending_delivery = {"alpha": [], "beta": []}
