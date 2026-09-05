"""
MCP Server - Exposes battle state and command tools for MCP clients.

This server allows MCP clients (like Claude Code) to:
- Read battle state via resources
- Issue ship commands via tools
- Communicate with enemy admiral
- Control battle flow (ready, surrender, draw)

Supports two modes:
- HTTP mode (--http): Connects to battle HTTP API for cross-process communication.
  This is the mode to use for anything real - MCP clients spawn the server as a
  separate process, so only HTTP can reach the running battle.
- Shared memory (no --http): Uses the in-process MCPSharedState singleton. This
  only works when the server is created *inside* the battle process (e.g. from a
  test or an embedding harness); launched standalone it sees an empty state and
  ready() has nothing to signal.
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
            # Full 3D position, kept so ranges are true ranges and not the
            # in-plane projection used for drawing.
            "pos3": (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)),
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
            "pos3": (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)),
            "vx": vel.get(axis_x, 0),
            "vy": vel.get(axis_y, 0),
            # Enemy hull is available via get_battle_state; the map keeps the
            # legend compact rather than modelling limited sensor information.
            "hull": None,
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

    # Show enemy ships. The map omits hull as a display choice - note that
    # get_battle_state DOES return exact enemy hull/armor, so this is not a
    # fog-of-war model.
    for s in enemy_list:
        arrow = s["arrow"]
        entry = f"  {arrow}[{s['label']}] {s['name'][:15]:<15} (enemy)"
        lines.append(f"║{entry:<62}║")

    # Calculate and show distances between closest pairs
    lines.append(f"║{'':>62}║")
    lines.append(f"║  DISTANCES:{'':>50}║")

    # Show distance from each friendly to closest enemy.
    # Ranges are true 3D separations - using only the two projected axes
    # understated out-of-plane ranges by orders of magnitude.
    for fs in friendly_list:
        min_dist = float('inf')
        min_plane_dist = 0.0
        closest_enemy = None
        for es in enemy_list:
            dx3 = fs["pos3"][0] - es["pos3"][0]
            dy3 = fs["pos3"][1] - es["pos3"][1]
            dz3 = fs["pos3"][2] - es["pos3"][2]
            dist = math.sqrt(dx3*dx3 + dy3*dy3 + dz3*dz3)
            if dist < min_dist:
                min_dist = dist
                dxp = fs["x"] - es["x"]
                dyp = fs["y"] - es["y"]
                min_plane_dist = math.sqrt(dxp*dxp + dyp*dyp)
                closest_enemy = es
        if closest_enemy:
            entry = f"  [{fs['label']}] → [{closest_enemy['label']}]: {min_dist:.1f} km"
            # Only mention the in-plane figure when it actually differs, so the
            # map stays readable for co-planar engagements.
            if abs(min_dist - min_plane_dist) > 0.05:
                entry += f" ({min_plane_dist:.1f} km in-plane)"
            lines.append(f"║{entry:<62}║")

    lines.append(f"╠{border}╣")
    scale_line = f"  Scale: 1 char ≈ {scale:.1f}km projected  |  Arrows = velocity"
    lines.append(f"║{scale_line:<62}║")
    hull_line = "  Friendly hull shown  |  Enemy hull via get_battle_state"
    lines.append(f"║{hull_line:<62}║")
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
        # MCPSharedState.signal_ready() is a silent no-op when no ready flag
        # exists, which is exactly what happens when this server runs in its own
        # process (the flags are created by the battle runner's set_event_loop).
        # Fail loudly instead of hanging the battle forever.
        if faction not in getattr(self._state, "_ready_flags", {}):
            raise RuntimeError(
                "No battle is running in this process. Shared-memory mode only works when "
                "the MCP server runs inside the battle process; start the battle with "
                "scripts/mcp_battle.py and connect this server with "
                "--http http://localhost:8765."
            )
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

    async def get_status_async(self) -> Dict[str, Any]:
        """Fetch battle status (checkpoint, who the runner is waiting for)."""
        return await self._client.get_status()

    async def get_draft_state_async(self) -> Dict[str, Any]:
        """Fetch the draft-phase state for this faction."""
        return await self._client.get_draft_state()

    async def draft_select_async(
        self, ships: List[Dict[str, Any]], rationale: str = ""
    ) -> Dict[str, Any]:
        """Submit a draft fleet selection."""
        return await self._client.draft_select(ships, rationale)

    async def draft_formation_async(
        self,
        placements: List[Dict[str, Any]],
        formation_name: str = "",
        rationale: str = "",
    ) -> Dict[str, Any]:
        """Submit draft formation placements."""
        return await self._client.draft_formation(
            placements, formation_name, rationale)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()


def describe_http_failure(base_url: str, exc: BaseException) -> str:
    """
    Turn an httpx transport error into something a commander can act on.

    Without this the agent sees a raw ConnectError/ReadTimeout traceback and
    cannot tell "battle not started" from "opposing side is thinking".
    """
    name = type(exc).__name__
    if "Timeout" in name:
        return (
            f"Battle server at {base_url} did not respond in time. It is most likely busy "
            f"resolving the opposing side's turn (synchronous LLM calls block the runner). "
            f"Wait a few seconds and retry."
        )
    if "Connect" in name or isinstance(exc, OSError):
        return (
            f"Battle server not reachable at {base_url}. Start it with: "
            f"uv run python scripts/mcp_battle.py --config <fleet_config.json>"
        )
    return f"Battle server error talking to {base_url}: {name}: {exc}"


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
                name="get_status",
                description=(
                    "Get battle flow status: current checkpoint number, whether the battle is "
                    "running, and which factions the runner is still waiting on for ready()"
                ),
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
                            "description": (
                                "WORLD-FRAME direction vector for HEADING maneuver (will be "
                                "normalized). Same coordinate system as position_km and "
                                "velocity_vector in get_battle_state - NOT relative to the "
                                "ship's nose. To fly at an enemy, use "
                                "(enemy.position_km - own.position_km)."
                            ),
                            "properties": {
                                "x": {"type": "number", "description": "World X component"},
                                "y": {"type": "number", "description": "World Y component"},
                                "z": {"type": "number", "description": "World Z component"}
                            },
                            "required": ["x", "y", "z"]
                        },
                    },
                    "required": ["ship_id", "maneuver_type"],
                },
            ),
            Tool(
                name="set_weapons_order",
                description=(
                    "Set weapons firing mode. Spinal weapons are high-damage forward-arc only. "
                    "Turret weapons have 180-degree arc but lower damage. Point defense is fully "
                    "automatic and cannot be commanded."
                ),
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
                        "max_range_km": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "description": "Required for FIRE_AT_RANGE: fire only when range <= this value (km)",
                        },
                        "min_hit_probability": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "For FIRE_WHEN_OPTIMAL: minimum hit probability to fire (0-1, default 0.3)",
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
                description=(
                    "Launch a torpedo from a ship. Requires a torpedo launcher with a ready "
                    "tube; the launch is reported as an error if the launcher is unavailable."
                ),
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
            Tool(
                name="get_draft_state",
                description=(
                    "Draft phase only: get your point budget, the costed ship "
                    "catalog, and your current picks. Draft flow: "
                    "get_draft_state -> select_fleet -> set_formation "
                    "(optional) -> ready() to commit."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="select_fleet",
                description=(
                    "Draft phase only: buy your fleet from the catalog (see "
                    "get_draft_state for hulls, costs and your budget). "
                    "Replaces any previous selection. Commit with ready()."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ship_type": {"type": "string"},
                                    "count": {"type": "integer", "minimum": 1},
                                },
                                "required": ["ship_type", "count"],
                            },
                            "description": "Hull classes and how many of each to buy",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "One or two sentences on the fleet concept",
                        },
                    },
                    "required": ["ships"],
                },
            ),
            Tool(
                name="set_formation",
                description=(
                    "Draft phase only: place your drafted ships. Offsets are "
                    "km from your fleet anchor in YOUR frame: +x points at "
                    "the enemy, y lateral, z vertical (limits in "
                    "get_draft_state). Unplaced ships keep default "
                    "line-abreast slots. May be called again to adjust."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "placements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ship_name": {"type": "string"},
                                    "x_km": {"type": "number"},
                                    "y_km": {"type": "number"},
                                    "z_km": {"type": "number"},
                                },
                                "required": ["ship_name", "x_km", "y_km"],
                            },
                        },
                        "formation_name": {
                            "type": "string",
                            "description": "Short name for the formation (e.g. 'PD wall')",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this shape",
                        },
                    },
                    "required": ["placements"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        # Get current timestamp from state.
        # Every HTTP interaction is guarded: an unguarded httpx error surfaces
        # to the agent as an opaque traceback with no way to recover.
        try:
            if is_http_mode:
                state_dict = await state_provider.get_state_dict_async()
                timestamp = state_dict.get("timestamp", 0.0)
            else:
                current_state = state.get_state(faction)
                timestamp = current_state.timestamp
                state_dict = None  # Lazy load if needed
        except Exception as e:
            return [TextContent(type="text", text=describe_http_failure(http_url, e))]

        async def add_command(command: MCPCommand) -> None:
            """Add command using appropriate provider."""
            if is_http_mode:
                await state_provider.send_command_async(command)
            else:
                state_provider.add_command(faction, command)

        def get_state_sync() -> Dict[str, Any]:
            """Return the state dict, fetching it lazily in shared-memory mode."""
            nonlocal state_dict
            if state_dict is None:
                state_dict = state_provider.get_state_dict(faction)
            return state_dict

        def validate_ship_id(ship_id: Optional[str]) -> Optional[str]:
            """Return an error message if ship_id is not one of ours, else None."""
            ships = get_state_sync().get("friendly_ships", [])
            if not ships:
                return None  # No state yet - let the runner report it
            valid = [s.get("ship_id") for s in ships]
            if ship_id not in valid:
                return f"Unknown friendly ship '{ship_id}'. Your ships: {valid}"
            return None

        def validate_target_id(target_id: Optional[str]) -> Optional[str]:
            """Return an error message if target_id is not a live enemy, else None."""
            enemies = get_state_sync().get("enemy_ships", [])
            if not enemies:
                return None
            valid = [s.get("ship_id") for s in enemies]
            if target_id not in valid:
                return f"Unknown enemy target '{target_id}'. Live enemies: {valid}"
            return None

        try:
            return await _dispatch_tool(
                name, arguments, timestamp, add_command,
                get_state_sync, validate_ship_id, validate_target_id,
            )
        except Exception as e:
            if is_http_mode:
                return [TextContent(type="text", text=describe_http_failure(http_url, e))]
            return [TextContent(type="text", text=f"Command failed: {type(e).__name__}: {e}")]

    async def _dispatch_tool(
        name,
        arguments,
        timestamp,
        add_command,
        get_state_sync,
        validate_ship_id,
        validate_target_id,
    ) -> List[TextContent]:
        if name == "get_battle_state":
            return [TextContent(
                type="text",
                text=json.dumps(get_state_sync(), indent=2),
            )]

        elif name == "get_status":
            if is_http_mode:
                status = await state_provider.get_status_async()
            else:
                snapshot = get_state_sync()
                status = {
                    "status": "running" if snapshot.get("is_battle_active") else "ended",
                    "checkpoint": snapshot.get("checkpoint_number", 0),
                    "waiting_for": [],
                    "connected_factions": [faction],
                }
            return [TextContent(type="text", text=json.dumps(status, indent=2))]

        elif name == "get_ship_status":
            ship_id = arguments.get("ship_id")
            for ship in get_state_sync().get("friendly_ships", []):
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
            err = validate_ship_id(arguments.get("ship_id"))
            if err:
                return [TextContent(type="text", text=f"ERROR: {err}")]
            maneuver_type = arguments.get("maneuver_type")
            valid_maneuvers = ["INTERCEPT", "EVASIVE", "BRAKE", "MAINTAIN", "PADLOCK", "HEADING"]
            if maneuver_type not in valid_maneuvers:
                return [TextContent(
                    type="text",
                    text=f"ERROR: Unknown maneuver_type '{maneuver_type}'. Valid: {valid_maneuvers}",
                )]
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
            err = validate_ship_id(arguments.get("ship_id"))
            if err:
                return [TextContent(type="text", text=f"ERROR: {err}")]
            valid_modes = ["FIRE_IMMEDIATE", "FIRE_WHEN_OPTIMAL", "FIRE_AT_RANGE", "HOLD_FIRE", "FREE_FIRE"]
            for key in ("spinal_mode", "turret_mode"):
                mode = arguments.get(key)
                if mode is not None and mode not in valid_modes:
                    return [TextContent(
                        type="text",
                        text=f"ERROR: Unknown {key} '{mode}'. Valid: {valid_modes}",
                    )]
                # FIRE_AT_RANGE without a range can never fire (range <= 0.0).
                if mode == "FIRE_AT_RANGE" and not (arguments.get("max_range_km") or 0) > 0:
                    return [TextContent(
                        type="text",
                        text=f"ERROR: {key}=FIRE_AT_RANGE requires max_range_km > 0",
                    )]
            command = MCPCommand(
                command_type=MCPCommandType.SET_WEAPONS_ORDER,
                ship_id=arguments.get("ship_id"),
                parameters={
                    "spinal_mode": arguments.get("spinal_mode"),
                    "turret_mode": arguments.get("turret_mode"),
                    "max_range_km": arguments.get("max_range_km"),
                    "min_hit_probability": arguments.get("min_hit_probability"),
                },
                timestamp=timestamp,
            )
            await add_command(command)
            return [TextContent(
                type="text",
                text=f"Weapons order set for {arguments.get('ship_id')}",
            )]

        elif name == "set_primary_target":
            err = validate_ship_id(arguments.get("ship_id"))
            if err:
                return [TextContent(type="text", text=f"ERROR: {err}")]
            target_id = arguments.get("target_id")
            if target_id != "NONE":
                # An unknown target id silently disables every gun on the ship,
                # so reject it here instead of confirming it back in state.
                err = validate_target_id(target_id)
                if err:
                    return [TextContent(type="text", text=f"ERROR: {err}")]
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
            err = validate_ship_id(arguments.get("ship_id")) or validate_target_id(arguments.get("target_id"))
            if err:
                return [TextContent(type="text", text=f"ERROR: {err}")]
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
            err = validate_ship_id(arguments.get("ship_id"))
            if err:
                return [TextContent(type="text", text=f"ERROR: {err}")]
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
                status = await state_provider.signal_ready_async()
            else:
                state_provider.signal_ready(faction)
                status = {"status": "ready", "faction": faction}
            # Return the server's status payload so the commander can see which
            # checkpoint it just committed to instead of a fixed string.
            return [TextContent(
                type="text",
                text="Ready signal sent - waiting for battle to advance.\n"
                     + json.dumps(status, indent=2),
            )]

        elif name == "battle_plot":
            state_dict = get_state_sync()
            projection = arguments.get("projection", "xy")
            plot = generate_battle_plot(state_dict, faction, projection)
            return [TextContent(
                type="text",
                text=plot,
            )]

        elif name == "get_draft_state":
            if not is_http_mode:
                return [TextContent(
                    type="text",
                    text="Drafting requires HTTP mode (start the server with --http).",
                )]
            draft_state = await state_provider.get_draft_state_async()
            if "error" in draft_state:
                return [TextContent(type="text", text=f"ERROR: {draft_state['error']}")]
            return [TextContent(type="text", text=json.dumps(draft_state, indent=2))]

        elif name == "select_fleet":
            if not is_http_mode:
                return [TextContent(
                    type="text",
                    text="Drafting requires HTTP mode (start the server with --http).",
                )]
            result = await state_provider.draft_select_async(
                arguments.get("ships"), arguments.get("rationale", ""))
            if result.get("status") != "ok":
                return [TextContent(
                    type="text",
                    text=f"ERROR: {result.get('error', json.dumps(result))}",
                )]
            roster = "\n".join(
                f"  - {s['ship_name']} ({s['ship_type']}, {s['cost']} pts)"
                for s in result.get("your_ships", []))
            return [TextContent(
                type="text",
                text=(
                    f"Fleet purchased: {result['points_spent']} pts spent, "
                    f"{result['points_remaining']} remaining.\n{roster}\n"
                    "Place it with set_formation (optional), then commit "
                    "with ready()."
                ),
            )]

        elif name == "set_formation":
            if not is_http_mode:
                return [TextContent(
                    type="text",
                    text="Drafting requires HTTP mode (start the server with --http).",
                )]
            result = await state_provider.draft_formation_async(
                arguments.get("placements"),
                arguments.get("formation_name", ""),
                arguments.get("rationale", ""),
            )
            if result.get("status") != "ok":
                return [TextContent(
                    type="text",
                    text=f"ERROR: {result.get('error', json.dumps(result))}",
                )]
            lines = [f"Formation '{result.get('formation_name', 'custom')}' set:"]
            for s in result.get("your_ships", []):
                off = s.get("offset_km", {})
                lines.append(
                    f"  - {s['ship_name']}: x={off.get('x', 0):+.0f} "
                    f"y={off.get('y', 0):+.0f} z={off.get('z', 0):+.0f} km")
            for note in result.get("notes", []):
                lines.append(f"  note: {note}")
            lines.append("Commit your draft with ready().")
            return [TextContent(type="text", text="\n".join(lines))]

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
        import sys
        # Loud, on stderr so it cannot corrupt the stdio JSON-RPC stream.
        print(
            "[mcp_server] WARNING: starting in shared-memory mode. This process has no "
            "battle in it, so state will be empty and ready() will fail. Pass "
            "--http http://localhost:8765 to talk to a running battle.",
            file=sys.stderr,
        )
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
