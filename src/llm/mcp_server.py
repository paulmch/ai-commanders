"""
MCP Server - Exposes battle state and command tools for MCP clients.

This server allows MCP clients (like Claude Code) to:
- Read battle state via resources
- Issue ship commands via tools
- Communicate with enemy admiral
- Control battle flow (ready, surrender, draw)

Supports two modes:
- Shared memory (default): Uses in-process singleton for local battles
- HTTP mode (--http): Connects to battle HTTP API for cross-process communication
"""

from __future__ import annotations

import json
import asyncio
import argparse
from typing import Dict, Any, List, Optional, Protocol

try:
    from mcp.server import Server
    from mcp.types import Resource, Tool, TextContent
    from mcp.server.stdio import stdio_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    Resource = None
    Tool = None
    TextContent = None
    stdio_server = None

from .mcp_state import get_mcp_state, MCPCommand, MCPCommandType, MCPHttpClient

import math


def get_velocity_arrow(vx: float, vy: float) -> str:
    """Convert velocity vector to arrow character."""
    if abs(vx) < 0.1 and abs(vy) < 0.1:
        return "·"  # Stationary

    angle = math.atan2(vy, vx)  # radians
    # 8 directions, starting from right (0 rad) going counterclockwise
    # Normalize angle to [0, 2*pi]
    if angle < 0:
        angle += 2 * math.pi
    # Each sector is pi/4 wide
    index = int((angle + math.pi / 8) / (math.pi / 4)) % 8
    arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
    return arrows[index]


def generate_battle_plot(
    state_dict: Dict[str, Any],
    faction: str,
    projection: str = "xy",
) -> str:
    """
    Generate ASCII tactical map showing ship positions and velocities.

    Args:
        state_dict: Battle state dictionary with friendly_ships and enemy_ships
        faction: Our faction ("alpha" or "beta")
        projection: Which 2D plane to project onto ("xy", "xz", "yz")

    Returns:
        ASCII tactical map string
    """
    # Extract ship data
    friendly_ships = state_dict.get("friendly_ships", [])
    enemy_ships = state_dict.get("enemy_ships", [])
    timestamp = state_dict.get("timestamp", 0)

    # Determine axis mapping based on projection
    if projection == "xy":
        axis_x, axis_y = "x", "y"
        axis_label = "X/Y plane (Z ignored)"
    elif projection == "xz":
        axis_x, axis_y = "x", "z"
        axis_label = "X/Z plane (Y ignored)"
    else:  # yz
        axis_x, axis_y = "y", "z"
        axis_label = "Y/Z plane (X ignored)"

    # Collect all ship data
    all_ships = []

    for ship in friendly_ships:
        pos = ship.get("position_km", {})
        vel = ship.get("velocity_vector", {})
        all_ships.append({
            "id": ship.get("ship_id", "?"),
            "name": ship.get("ship_name", ship.get("ship_id", "?")),
            "x": pos.get(axis_x, 0),
            "y": pos.get(axis_y, 0),
            "vx": vel.get(axis_x, 0),
            "vy": vel.get(axis_y, 0),
            "hull": ship.get("hull_integrity"),  # 0-100
            "friendly": True,
        })

    for ship in enemy_ships:
        pos = ship.get("position_km", {})
        vel = ship.get("velocity_vector", {})
        all_ships.append({
            "id": ship.get("ship_id", "?"),
            "name": ship.get("ship_name", ship.get("ship_id", "?")),
            "x": pos.get(axis_x, 0),
            "y": pos.get(axis_y, 0),
            "vx": vel.get(axis_x, 0),
            "vy": vel.get(axis_y, 0),
            "hull": None,  # Fog of war - no hull for enemies
            "friendly": False,
        })

    if not all_ships:
        return "No ships to display."

    # Calculate bounds
    xs = [s["x"] for s in all_ships]
    ys = [s["y"] for s in all_ships]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add padding (10% or minimum 10km)
    padding_x = max((max_x - min_x) * 0.1, 10)
    padding_y = max((max_y - min_y) * 0.1, 10)
    min_x -= padding_x
    max_x += padding_x
    min_y -= padding_y
    max_y += padding_y

    # Ensure non-zero range
    if max_x - min_x < 1:
        min_x -= 5
        max_x += 5
    if max_y - min_y < 1:
        min_y -= 5
        max_y += 5

    # Grid size (~60 chars wide, ~20 chars tall for the plot area)
    grid_width = 60
    grid_height = 20

    # Calculate scale
    scale_x = (max_x - min_x) / grid_width
    scale_y = (max_y - min_y) / grid_height
    scale = max(scale_x, scale_y)  # Use same scale for both to preserve aspect

    # Create grid (y=0 is top of display)
    grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

    # Ship labels for the map
    friendly_label = "A" if faction == "alpha" else "B"
    enemy_label = "B" if faction == "alpha" else "A"

    # Place ships on grid
    placed_ships = []
    friendly_idx = 1
    enemy_idx = 1

    for ship in all_ships:
        # Convert to grid coordinates
        gx = int((ship["x"] - min_x) / scale) if scale > 0 else grid_width // 2
        gy = int((max_y - ship["y"]) / scale) if scale > 0 else grid_height // 2  # Flip Y

        # Clamp to grid bounds
        gx = max(0, min(grid_width - 1, gx))
        gy = max(0, min(grid_height - 1, gy))

        # Assign label
        if ship["friendly"]:
            label = f"{friendly_label}{friendly_idx}"
            friendly_idx += 1
        else:
            label = f"{enemy_label}{enemy_idx}"
            enemy_idx += 1

        # Get velocity arrow
        arrow = get_velocity_arrow(ship["vx"], ship["vy"])

        placed_ships.append({
            **ship,
            "gx": gx,
            "gy": gy,
            "label": label,
            "arrow": arrow,
        })

    # Draw ships on grid - use markers that won't overlap
    for ps in placed_ships:
        gx, gy = ps["gx"], ps["gy"]
        marker = f"[{ps['label']}]"
        # Try to place the marker (5 chars wide)
        start_x = max(0, min(gx - 2, grid_width - 5))
        if gy < grid_height:
            for i, ch in enumerate(marker):
                if start_x + i < grid_width:
                    grid[gy][start_x + i] = ch

    # Build output
    lines = []
    border = "═" * 62
    lines.append(f"╔{border}╗")
    lines.append(f"║  TACTICAL MAP - T={timestamp:.0f}s  ({axis_label}){'':>14}║")
    lines.append(f"╠{border}╣")

    # Add grid rows
    for row in grid:
        line = "".join(row)
        lines.append(f"║ {line} ║")

    lines.append(f"╠{border}╣")

    # Legend section
    lines.append(f"║  SHIPS:{'':>54}║")

    # Group by friendly/enemy
    friendly_list = [s for s in placed_ships if s["friendly"]]
    enemy_list = [s for s in placed_ships if not s["friendly"]]

    # Show friendly ships with hull
    for s in friendly_list:
        hull_str = f"({s['hull']:.0f}%)" if s["hull"] is not None else ""
        arrow = s["arrow"]
        entry = f"  {arrow}[{s['label']}] {s['name'][:15]:<15} {hull_str}"
        lines.append(f"║{entry:<62}║")

    # Show enemy ships (no hull - fog of war)
    for s in enemy_list:
        arrow = s["arrow"]
        entry = f"  {arrow}[{s['label']}] {s['name'][:15]:<15} (enemy)"
        lines.append(f"║{entry:<62}║")

    # Calculate and show distances between closest pairs
    lines.append(f"║{'':>62}║")
    lines.append(f"║  DISTANCES:{'':>50}║")

    # Show distance from each friendly to closest enemy
    for fs in friendly_list:
        min_dist = float('inf')
        closest_enemy = None
        for es in enemy_list:
            dx = fs["x"] - es["x"]
            dy = fs["y"] - es["y"]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = es
        if closest_enemy:
            entry = f"  [{fs['label']}] → [{closest_enemy['label']}]: {min_dist:.1f} km"
            lines.append(f"║{entry:<62}║")

    lines.append(f"╠{border}╣")
    lines.append(f"║  Scale: 1 char ≈ {scale:.1f}km  |  Arrows show velocity direction{'':>5}║")
    lines.append(f"║  Friendly hull shown  |  Enemy hull hidden (fog of war){'':>6}║")
    lines.append(f"╚{border}╝")

    return "\n".join(lines)


class StateProvider(Protocol):
    """Protocol for state providers (shared memory or HTTP)."""

    def get_state_dict(self, faction: str) -> Dict[str, Any]:
        """Get battle state as dictionary."""
        ...

    def add_command(self, faction: str, command: MCPCommand) -> None:
        """Add a command for processing."""
        ...

    def signal_ready(self, faction: str) -> None:
        """Signal that commands are complete."""
        ...


class SharedStateProvider:
    """State provider using in-process shared memory."""

    def __init__(self, faction: str):
        self.faction = faction
        self._state = get_mcp_state()
        self._state.register_faction(faction)

    def get_state_dict(self, faction: str) -> Dict[str, Any]:
        return self._state.get_state_dict(faction)

    def add_command(self, faction: str, command: MCPCommand) -> None:
        self._state.add_command(faction, command)

    def signal_ready(self, faction: str) -> None:
        self._state.signal_ready(faction)


class HttpStateProvider:
    """State provider using HTTP API."""

    def __init__(self, base_url: str, faction: str):
        self.faction = faction
        self.base_url = base_url
        self._client = MCPHttpClient(base_url, faction)
        self._pending_commands: List[MCPCommand] = []

    async def get_state_dict_async(self) -> Dict[str, Any]:
        """Async version of get_state_dict."""
        return await self._client.get_state_dict()

    def get_state_dict(self, faction: str) -> Dict[str, Any]:
        """Sync wrapper - runs async in event loop."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create a future
            future = asyncio.ensure_future(self._client.get_state_dict())
            return asyncio.get_event_loop().run_until_complete(future)
        else:
            return loop.run_until_complete(self._client.get_state_dict())

    def add_command(self, faction: str, command: MCPCommand) -> None:
        """Queue command for later sending."""
        self._pending_commands.append(command)

    async def send_command_async(self, command: MCPCommand) -> Dict[str, Any]:
        """Send a single command via HTTP."""
        return await self._client.send_command(command)

    def signal_ready(self, faction: str) -> None:
        """Sync wrapper for signal_ready."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(self._client.signal_ready())
        else:
            loop.run_until_complete(self._client.signal_ready())

    async def signal_ready_async(self) -> Dict[str, Any]:
        """Async version of signal_ready."""
        return await self._client.signal_ready()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()


def create_mcp_server(
    faction: str,
    http_url: Optional[str] = None,
) -> Optional['Server']:
    """
    Create an MCP server for a specific faction.

    Args:
        faction: "alpha" or "beta"
        http_url: If provided, use HTTP mode to connect to this URL

    Returns:
        MCP Server instance or None if MCP not available
    """
    if not MCP_AVAILABLE:
        return None

    server = Server(f"ai-commanders-{faction}")

    # Create state provider (HTTP or shared memory)
    if http_url:
        state_provider = HttpStateProvider(http_url, faction)
        is_http_mode = True
    else:
        state_provider = SharedStateProvider(faction)
        is_http_mode = False

    # For backwards compatibility, also get shared state reference
    state = get_mcp_state() if not is_http_mode else None

    # === RESOURCES ===

    @server.list_resources()
    async def list_resources() -> List[Resource]:
        """List available resources."""
        return [
            Resource(
                uri=f"battle://state/{faction}",
                name="Battle State",
                description="Current battle snapshot with friendly ships, enemy ships, and projectiles",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a resource by URI."""
        if uri == f"battle://state/{faction}":
            if is_http_mode:
                state_dict = await state_provider.get_state_dict_async()
            else:
                state_dict = state_provider.get_state_dict(faction)
            return json.dumps(state_dict, indent=2)
        raise ValueError(f"Unknown resource: {uri}")

    # === TOOLS ===

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        return [
            Tool(
                name="get_battle_state",
                description="Get current battle state snapshot",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="get_ship_status",
                description="Get detailed status for a specific friendly ship",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ship_id": {
                            "type": "string",
                            "description": "ID of the ship to query",
                        },
                    },
                    "required": ["ship_id"],
                },
            ),
            Tool(
                name="set_maneuver",
                description="Set ship maneuver (movement control). INTERCEPT: burn toward target. EVASIVE: random evasive pattern. BRAKE: flip and decelerate. MAINTAIN: coast at current velocity. PADLOCK: coast while tracking target with nose. HEADING: fly in specific 3D direction.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ship_id": {
                            "type": "string",
                            "description": "ID of the ship",
                        },
                        "maneuver_type": {
                            "type": "string",
                            "enum": ["INTERCEPT", "EVASIVE", "BRAKE", "MAINTAIN", "PADLOCK", "HEADING"],
                            "description": "Type of maneuver: INTERCEPT (approach target), EVASIVE (dodge), BRAKE (slow down), MAINTAIN (coast), PADLOCK (track target, no thrust), HEADING (fly in direction)",
                        },
                        "throttle": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Throttle setting (0-1)",
                        },
                        "target_id": {
                            "type": "string",
                            "description": "Target ship ID (required for INTERCEPT, EVASIVE, PADLOCK)",
                        },
                        "heading_direction": {
                            "type": "object",
                            "description": "Direction vector for HEADING maneuver (will be normalized)",
                            "properties": {
                                "x": {"type": "number", "description": "Forward/backward (+forward toward enemy, -backward)"},
                                "y": {"type": "number", "description": "Left/right (+right, -left)"},
                                "z": {"type": "number", "description": "Up/down (+up, -down)"}
                            },
                            "required": ["x", "y", "z"]
                        },
                    },
                    "required": ["ship_id", "maneuver_type"],
                },
            ),
            Tool(
                name="set_weapons_order",
                description="Set weapons firing mode. Spinal weapons are high-damage forward-arc only. Turret weapons have 180-degree arc but lower damage.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ship_id": {
                            "type": "string",
                            "description": "ID of the ship",
                        },
                        "spinal_mode": {
                            "type": "string",
                            "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE", "FREE_FIRE"],
                            "description": "Spinal weapon mode: FIRE_IMMEDIATE (fire when ready), FIRE_WHEN_OPTIMAL (fire when hit prob >= threshold), FIRE_AT_RANGE (fire within range), HOLD_FIRE (don't fire), FREE_FIRE (fire at any valid target)",
                        },
                        "turret_mode": {
                            "type": "string",
                            "enum": ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE", "FREE_FIRE"],
                            "description": "Turret weapon mode: same options as spinal",
                        },
                        "pd_mode": {
                            "type": "string",
                            "enum": ["AUTO", "MANUAL", "OFF"],
                            "description": "Point defense mode for intercepting missiles/torpedoes",
                        },
                    },
                    "required": ["ship_id"],
                },
            ),
            Tool(
                name="set_primary_target",
                description="Set primary target for a ship",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ship_id": {
                            "type": "string",
                            "description": "ID of the ship",
                        },
                        "target_id": {
                            "type": "string",
                            "description": "ID of the enemy ship to target (or 'NONE' to clear)",
                        },
                    },
                    "required": ["ship_id", "target_id"],
                },
            ),
            Tool(
                name="launch_torpedo",
                description="Launch a torpedo from a ship",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ship_id": {
                            "type": "string",
                            "description": "ID of the ship",
                        },
                        "target_id": {
                            "type": "string",
                            "description": "ID of the target ship",
                        },
                    },
                    "required": ["ship_id", "target_id"],
                },
            ),
            Tool(
                name="set_radiators",
                description="Extend or retract radiators",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ship_id": {
                            "type": "string",
                            "description": "ID of the ship",
                        },
                        "extend": {
                            "type": "boolean",
                            "description": "True to extend, False to retract",
                        },
                    },
                    "required": ["ship_id", "extend"],
                },
            ),
            Tool(
                name="send_message",
                description="Send a message to enemy admiral (max 3 per turn)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Message content",
                        },
                        "recipient": {
                            "type": "string",
                            "enum": ["enemy", "all"],
                            "description": "Message recipient",
                        },
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="propose_fleet_draw",
                description="Propose a draw for the entire fleet",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="accept_fleet_draw",
                description="Accept enemy's draw proposal",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="surrender_fleet",
                description="Surrender the entire fleet",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="ready",
                description="Signal that all commands for this turn have been issued",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="battle_plot",
                description="Generate ASCII tactical map showing ship positions and velocities",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "projection": {
                            "type": "string",
                            "enum": ["xy", "xz", "yz"],
                            "description": "Which 2D plane to project onto (default: xy)",
                        },
                    },
                    "required": [],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        # Get current timestamp from state
        if is_http_mode:
            state_dict = await state_provider.get_state_dict_async()
            timestamp = state_dict.get("timestamp", 0.0)
        else:
            current_state = state.get_state(faction)
            timestamp = current_state.timestamp
            state_dict = None  # Lazy load if needed

        async def add_command(command: MCPCommand) -> None:
            """Add command using appropriate provider."""
            if is_http_mode:
                await state_provider.send_command_async(command)
            else:
                state_provider.add_command(faction, command)

        if name == "get_battle_state":
            if state_dict is None:
                state_dict = state_provider.get_state_dict(faction)
            return [TextContent(
                type="text",
                text=json.dumps(state_dict, indent=2),
            )]

        elif name == "get_ship_status":
            ship_id = arguments.get("ship_id")
            if state_dict is None:
                state_dict = state_provider.get_state_dict(faction)
            for ship in state_dict.get("friendly_ships", []):
                if ship.get("ship_id") == ship_id:
                    return [TextContent(
                        type="text",
                        text=json.dumps(ship, indent=2),
                    )]
            return [TextContent(
                type="text",
                text=f"Ship {ship_id} not found in friendly fleet",
            )]

        elif name == "set_maneuver":
            command = MCPCommand(
                command_type=MCPCommandType.SET_MANEUVER,
                ship_id=arguments.get("ship_id"),
                parameters={
                    "maneuver_type": arguments.get("maneuver_type"),
                    "throttle": arguments.get("throttle", 1.0),
                    "target_id": arguments.get("target_id"),
                    "heading_direction": arguments.get("heading_direction"),
                },
                timestamp=timestamp,
            )
            await add_command(command)
            heading_info = ""
            if arguments.get("heading_direction"):
                hd = arguments.get("heading_direction")
                heading_info = f" direction=({hd.get('x', 0)}, {hd.get('y', 0)}, {hd.get('z', 0)})"
            return [TextContent(
                type="text",
                text=f"Maneuver set for {arguments.get('ship_id')}: {arguments.get('maneuver_type')}{heading_info}",
            )]

        elif name == "set_weapons_order":
            command = MCPCommand(
                command_type=MCPCommandType.SET_WEAPONS_ORDER,
                ship_id=arguments.get("ship_id"),
                parameters={
                    "spinal_mode": arguments.get("spinal_mode"),
                    "turret_mode": arguments.get("turret_mode"),
                    "pd_mode": arguments.get("pd_mode"),
                },
                timestamp=timestamp,
            )
            await add_command(command)
            return [TextContent(
                type="text",
                text=f"Weapons order set for {arguments.get('ship_id')}",
            )]

        elif name == "set_primary_target":
            command = MCPCommand(
                command_type=MCPCommandType.SET_PRIMARY_TARGET,
                ship_id=arguments.get("ship_id"),
                parameters={
                    "target_id": arguments.get("target_id"),
                },
                timestamp=timestamp,
            )
            await add_command(command)
            return [TextContent(
                type="text",
                text=f"Primary target set for {arguments.get('ship_id')}: {arguments.get('target_id')}",
            )]

        elif name == "launch_torpedo":
            command = MCPCommand(
                command_type=MCPCommandType.LAUNCH_TORPEDO,
                ship_id=arguments.get("ship_id"),
                parameters={
                    "target_id": arguments.get("target_id"),
                },
                timestamp=timestamp,
            )
            await add_command(command)
            return [TextContent(
                type="text",
                text=f"Torpedo launch ordered for {arguments.get('ship_id')} at {arguments.get('target_id')}",
            )]

        elif name == "set_radiators":
            command = MCPCommand(
                command_type=MCPCommandType.SET_RADIATORS,
                ship_id=arguments.get("ship_id"),
                parameters={
                    "extend": arguments.get("extend"),
                },
                timestamp=timestamp,
            )
            await add_command(command)
            action = "Extended" if arguments.get("extend") else "Retracted"
            return [TextContent(
                type="text",
                text=f"{action} radiators for {arguments.get('ship_id')}",
            )]

        elif name == "send_message":
            command = MCPCommand(
                command_type=MCPCommandType.SEND_MESSAGE,
                parameters={
                    "content": arguments.get("content"),
                    "recipient": arguments.get("recipient", "enemy"),
                },
                timestamp=timestamp,
            )
            await add_command(command)
            return [TextContent(
                type="text",
                text=f"Message queued: {arguments.get('content')[:50]}...",
            )]

        elif name == "propose_fleet_draw":
            command = MCPCommand(
                command_type=MCPCommandType.PROPOSE_DRAW,
                parameters={},
                timestamp=timestamp,
            )
            await add_command(command)
            return [TextContent(
                type="text",
                text="Draw proposal submitted",
            )]

        elif name == "accept_fleet_draw":
            command = MCPCommand(
                command_type=MCPCommandType.PROPOSE_DRAW,
                parameters={"accept": True},
                timestamp=timestamp,
            )
            await add_command(command)
            return [TextContent(
                type="text",
                text="Draw acceptance submitted",
            )]

        elif name == "surrender_fleet":
            command = MCPCommand(
                command_type=MCPCommandType.SURRENDER,
                parameters={},
                timestamp=timestamp,
            )
            await add_command(command)
            return [TextContent(
                type="text",
                text="Fleet surrender submitted",
            )]

        elif name == "ready":
            command = MCPCommand(
                command_type=MCPCommandType.READY,
                parameters={},
                timestamp=timestamp,
            )
            await add_command(command)
            if is_http_mode:
                await state_provider.signal_ready_async()
            else:
                state_provider.signal_ready(faction)
            return [TextContent(
                type="text",
                text="Ready signal sent - waiting for battle to advance",
            )]

        elif name == "battle_plot":
            if state_dict is None:
                if is_http_mode:
                    state_dict = await state_provider.get_state_dict_async()
                else:
                    state_dict = state_provider.get_state_dict(faction)
            projection = arguments.get("projection", "xy")
            plot = generate_battle_plot(state_dict, faction, projection)
            return [TextContent(
                type="text",
                text=plot,
            )]

        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}",
            )]

    return server


async def run_mcp_server(faction: str, http_url: Optional[str] = None) -> None:
    """
    Run the MCP server for a faction.

    Args:
        faction: "alpha" or "beta"
        http_url: If provided, use HTTP mode to connect to this URL
    """
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP package not installed. Run: pip install mcp")

    server = create_mcp_server(faction, http_url=http_url)
    if server is None:
        raise RuntimeError("Failed to create MCP server")

    # Only register faction if using shared memory mode
    if http_url is None:
        state = get_mcp_state()
        state.register_faction(faction)

    # Import initialization options
    from mcp.server import InitializationOptions, NotificationOptions

    notification_options = NotificationOptions(
        prompts_changed=False,
        resources_changed=False,
        tools_changed=False,
    )

    init_options = InitializationOptions(
        server_name=f"ai-commanders-{faction}",
        server_version="1.0.0",
        capabilities=server.get_capabilities(
            notification_options=notification_options,
            experimental_capabilities={},
        ),
    )

    # Run with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


def main_alpha():
    """Entry point for alpha faction MCP server."""
    asyncio.run(run_mcp_server("alpha"))


def main_beta():
    """Entry point for beta faction MCP server."""
    asyncio.run(run_mcp_server("beta"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Server for AI Commanders battle control"
    )
    parser.add_argument(
        "--faction",
        choices=["alpha", "beta"],
        default="alpha",
        help="Faction to control (default: alpha)",
    )
    parser.add_argument(
        "--http",
        metavar="URL",
        help="HTTP API URL (e.g., http://localhost:8765). If provided, uses HTTP mode instead of shared memory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_mcp_server(args.faction, http_url=args.http))
