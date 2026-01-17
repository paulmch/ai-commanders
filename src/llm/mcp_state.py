"""
MCP Shared State - Thread-safe singleton for MCP server/controller communication.

Manages:
- Current battle state for MCP clients to read
- Pending commands from MCP tools
- Synchronization primitives for async coordination
"""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class MCPCommandType(Enum):
    """Types of commands that can be issued via MCP."""
    SET_MANEUVER = "set_maneuver"
    SET_WEAPONS_ORDER = "set_weapons_order"
    SET_PRIMARY_TARGET = "set_primary_target"
    LAUNCH_TORPEDO = "launch_torpedo"
    SET_RADIATORS = "set_radiators"
    SEND_MESSAGE = "send_message"
    PROPOSE_DRAW = "propose_draw"
    SURRENDER = "surrender"
    READY = "ready"


@dataclass
class MCPCommand:
    """A command issued via MCP tools."""
    command_type: MCPCommandType
    ship_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class MCPBattleState:
    """
    Current battle state for MCP consumption.

    Provides the same information that captains and admirals see:
    - Friendly ships: full status (armor, modules, weapons, combat stats)
    - Enemy ships: observable data (hull, armor damage, hit chance, combat stats)
    - Projectiles: incoming with bearing, weapon type, ETA
    - Torpedoes: active torpedo threats
    """
    timestamp: float = 0.0
    faction: str = ""

    # Friendly ships (full info - same data captains see about their own ship)
    friendly_ships: List[Dict[str, Any]] = field(default_factory=list)

    # Enemy ships (observable - same data captains see about enemies)
    enemy_ships: List[Dict[str, Any]] = field(default_factory=list)

    # Projectiles in flight (with bearing, weapon type, ETA)
    projectiles: List[Dict[str, Any]] = field(default_factory=list)

    # Torpedo threats (active torpedoes targeting our ships)
    torpedoes: List[Dict[str, Any]] = field(default_factory=list)

    # Chat history (recent messages)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)

    # Fleet summary
    fleet_summary: str = ""

    # Battle status
    is_battle_active: bool = False
    checkpoint_number: int = 0

    # Pending draw proposal from enemy
    enemy_proposed_draw: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "faction": self.faction,
            "friendly_ships": self.friendly_ships,
            "enemy_ships": self.enemy_ships,
            "projectiles": self.projectiles,
            "torpedoes": self.torpedoes,
            "chat_history": self.chat_history,
            "fleet_summary": self.fleet_summary,
            "is_battle_active": self.is_battle_active,
            "checkpoint_number": self.checkpoint_number,
            "enemy_proposed_draw": self.enemy_proposed_draw,
        }


class MCPSharedState:
    """
    Thread-safe singleton managing shared state between MCP server and controller.

    The MCP server reads state and writes commands.
    The battle controller reads commands and writes state.
    """

    _instance: Optional['MCPSharedState'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Thread-safe state storage
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()

        # Current battle state per faction
        self._battle_states: Dict[str, MCPBattleState] = {
            "alpha": MCPBattleState(faction="alpha"),
            "beta": MCPBattleState(faction="beta"),
        }

        # Pending commands per faction (cleared after processing)
        self._pending_commands: Dict[str, List[MCPCommand]] = {
            "alpha": [],
            "beta": [],
        }

        # Ready flags - set when MCP client signals commands complete
        self._ready_flags: Dict[str, asyncio.Event] = {}

        # Event loop reference for async coordination
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Battle registration
        self._active_factions: set = set()

        self._initialized = True

    @classmethod
    def get_instance(cls) -> 'MCPSharedState':
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._initialized = False
                cls._instance = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for async coordination."""
        self._loop = loop
        # Create events on the loop
        for faction in ["alpha", "beta"]:
            self._ready_flags[faction] = asyncio.Event()

    def register_faction(self, faction: str) -> None:
        """Register a faction as MCP-controlled."""
        with self._state_lock:
            self._active_factions.add(faction)
            self._battle_states[faction] = MCPBattleState(faction=faction)
            self._pending_commands[faction] = []

    def unregister_faction(self, faction: str) -> None:
        """Unregister a faction."""
        with self._state_lock:
            self._active_factions.discard(faction)

    def is_faction_active(self, faction: str) -> bool:
        """Check if a faction is MCP-controlled."""
        with self._state_lock:
            return faction in self._active_factions

    # === State Management (written by controller, read by MCP server) ===

    def update_state(self, faction: str, state: MCPBattleState) -> None:
        """Update battle state for a faction."""
        with self._state_lock:
            self._battle_states[faction] = state

    def get_state(self, faction: str) -> MCPBattleState:
        """Get current battle state for a faction."""
        with self._state_lock:
            return self._battle_states.get(faction, MCPBattleState(faction=faction))

    def get_state_dict(self, faction: str) -> Dict[str, Any]:
        """Get current battle state as dictionary."""
        return self.get_state(faction).to_dict()

    # === Command Management (written by MCP server, read by controller) ===

    def add_command(self, faction: str, command: MCPCommand) -> None:
        """Add a command from MCP client."""
        with self._command_lock:
            if faction in self._pending_commands:
                self._pending_commands[faction].append(command)

    def get_pending_commands(self, faction: str) -> List[MCPCommand]:
        """Get and clear pending commands for a faction."""
        with self._command_lock:
            commands = self._pending_commands.get(faction, [])
            self._pending_commands[faction] = []
            return commands

    def peek_pending_commands(self, faction: str) -> List[MCPCommand]:
        """Get pending commands without clearing."""
        with self._command_lock:
            return list(self._pending_commands.get(faction, []))

    # === Ready Signaling ===

    def signal_ready(self, faction: str) -> None:
        """Signal that MCP client has finished issuing commands for this turn."""
        if faction in self._ready_flags:
            self._ready_flags[faction].set()

    def clear_ready(self, faction: str) -> None:
        """Clear ready flag for next turn."""
        if faction in self._ready_flags:
            self._ready_flags[faction].clear()

    async def wait_for_ready(self, faction: str, timeout: float = 60.0) -> bool:
        """
        Wait for MCP client to signal ready.

        Args:
            faction: Faction to wait for
            timeout: Maximum wait time in seconds

        Returns:
            True if ready received, False if timeout
        """
        if faction not in self._ready_flags:
            return False

        try:
            await asyncio.wait_for(
                self._ready_flags[faction].wait(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

    def is_ready(self, faction: str) -> bool:
        """Check if faction is ready (non-blocking)."""
        if faction not in self._ready_flags:
            return False
        return self._ready_flags[faction].is_set()


# Convenience function to get singleton
def get_mcp_state() -> MCPSharedState:
    """Get the MCP shared state singleton."""
    return MCPSharedState.get_instance()


class MCPHttpClient:
    """
    HTTP client for MCP servers to communicate with the battle API server.

    Used when MCP servers run as separate processes (e.g., spawned by Claude Code)
    and need to communicate with the battle runner via HTTP.
    """

    def __init__(self, base_url: str, faction: str):
        """
        Initialize HTTP client.

        Args:
            base_url: Base URL of the battle API server (e.g., "http://localhost:8765")
            faction: Faction this client represents ("alpha" or "beta")
        """
        self.base_url = base_url.rstrip("/")
        self.faction = faction
        self._client = None

    async def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(timeout=30.0)
            except ImportError:
                raise RuntimeError("httpx package not installed. Run: pip install httpx")
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_state(self) -> MCPBattleState:
        """
        Fetch current battle state from the server.

        Returns:
            MCPBattleState for this faction
        """
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/state/{self.faction}")
        response.raise_for_status()

        data = response.json()
        return MCPBattleState(
            timestamp=data.get("timestamp", 0.0),
            faction=data.get("faction", self.faction),
            friendly_ships=data.get("friendly_ships", []),
            enemy_ships=data.get("enemy_ships", []),
            projectiles=data.get("projectiles", []),
            torpedoes=data.get("torpedoes", []),
            chat_history=data.get("chat_history", []),
            fleet_summary=data.get("fleet_summary", ""),
            is_battle_active=data.get("is_battle_active", False),
            checkpoint_number=data.get("checkpoint_number", 0),
            enemy_proposed_draw=data.get("enemy_proposed_draw", False),
        )

    async def get_state_dict(self) -> Dict[str, Any]:
        """
        Fetch current battle state as dictionary.

        Returns:
            Battle state as dictionary
        """
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/state/{self.faction}")
        response.raise_for_status()
        return response.json()

    async def send_command(self, command: MCPCommand) -> Dict[str, Any]:
        """
        Send a command to the battle server.

        Args:
            command: Command to send

        Returns:
            Server response
        """
        client = await self._get_client()

        # Serialize command
        command_data = {
            "command_type": command.command_type.value,
            "ship_id": command.ship_id,
            "parameters": command.parameters,
            "timestamp": command.timestamp,
        }

        response = await client.post(
            f"{self.base_url}/commands/{self.faction}",
            json=command_data,
        )
        response.raise_for_status()
        return response.json()

    async def send_commands(self, commands: List[MCPCommand]) -> Dict[str, Any]:
        """
        Send multiple commands to the battle server.

        Args:
            commands: List of commands to send

        Returns:
            Server response
        """
        client = await self._get_client()

        # Serialize commands
        commands_data = [
            {
                "command_type": cmd.command_type.value,
                "ship_id": cmd.ship_id,
                "parameters": cmd.parameters,
                "timestamp": cmd.timestamp,
            }
            for cmd in commands
        ]

        response = await client.post(
            f"{self.base_url}/commands/{self.faction}",
            json={"commands": commands_data},
        )
        response.raise_for_status()
        return response.json()

    async def signal_ready(self) -> Dict[str, Any]:
        """
        Signal that all commands for this turn have been issued.

        Returns:
            Server response
        """
        client = await self._get_client()
        response = await client.post(f"{self.base_url}/ready/{self.faction}")
        response.raise_for_status()
        return response.json()

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current battle status.

        Returns:
            Status dict with keys: status, checkpoint, waiting_for, etc.
        """
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> bool:
        """
        Check if the battle server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
