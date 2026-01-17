"""
MCP HTTP Server - REST API for MCP clients to communicate with the battle runner.

Provides HTTP endpoints for:
- Getting battle state per faction
- Submitting commands
- Signaling ready for turn advancement
- Querying battle status

This allows MCP servers (spawned by Claude Code as subprocesses) to communicate
with the battle runner even though they run in separate processes.
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Awaitable, TYPE_CHECKING

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

from .mcp_state import (
    MCPSharedState,
    MCPBattleState,
    MCPCommand,
    MCPCommandType,
    get_mcp_state,
)

if TYPE_CHECKING:
    from aiohttp import web as web_typing
    from .mcp_controller import MCPController


class MCPHttpServer:
    """
    HTTP API server for MCP client communication.

    Runs alongside the battle runner and provides REST endpoints for
    MCP servers to get state, send commands, and signal readiness.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        shared_state: Optional[MCPSharedState] = None,
    ):
        """
        Initialize HTTP server.

        Args:
            host: Host to bind to
            port: Port to listen on
            shared_state: Shared state instance (uses singleton if not provided)
        """
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp package not installed. Run: pip install aiohttp")

        self.host = host
        self.port = port
        self._state = shared_state or get_mcp_state()

        # Track which factions have connected
        self._connected_factions: set = set()

        # Battle status
        self._battle_status = "waiting"  # "waiting", "running", "paused", "ended"
        self._current_checkpoint = 0
        self._waiting_for: List[str] = []

        # Controllers for building state
        self._controllers: Dict[str, 'MCPController'] = {}

        # Callbacks
        self._on_ready_callback: Optional[Callable[[str], Awaitable[None]]] = None

        # Server instances
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    def register_controller(self, faction: str, controller: 'MCPController') -> None:
        """Register an MCP controller for a faction."""
        self._controllers[faction] = controller

    def set_on_ready_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Set callback to be called when a faction signals ready."""
        self._on_ready_callback = callback

    def set_battle_status(
        self,
        status: str,
        checkpoint: int = 0,
        waiting_for: Optional[List[str]] = None,
    ) -> None:
        """Update battle status."""
        self._battle_status = status
        self._current_checkpoint = checkpoint
        self._waiting_for = waiting_for or []

    async def start(self) -> None:
        """Start the HTTP server."""
        self._app = web.Application()
        self._setup_routes()

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        print(f"MCP HTTP API running on http://{self.host}:{self.port}")
        print(f"Waiting for MCP clients to connect...")

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

    def _setup_routes(self) -> None:
        """Setup HTTP routes."""
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/status", self._handle_status)
        self._app.router.add_get("/state/{faction}", self._handle_get_state)
        self._app.router.add_post("/commands/{faction}", self._handle_commands)
        self._app.router.add_post("/ready/{faction}", self._handle_ready)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok"})

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Get battle status."""
        return web.json_response({
            "status": self._battle_status,
            "checkpoint": self._current_checkpoint,
            "waiting_for": self._waiting_for,
            "connected_factions": list(self._connected_factions),
        })

    async def _handle_get_state(self, request: web.Request) -> web.Response:
        """Get battle state for a faction."""
        faction = request.match_info["faction"]

        if faction not in ("alpha", "beta"):
            return web.json_response(
                {"error": f"Invalid faction: {faction}"},
                status=400,
            )

        # Track connection
        self._connected_factions.add(faction)

        # Get state from shared state
        state = self._state.get_state(faction)
        return web.json_response(state.to_dict())

    async def _handle_commands(self, request: web.Request) -> web.Response:
        """Handle command submission."""
        faction = request.match_info["faction"]

        if faction not in ("alpha", "beta"):
            return web.json_response(
                {"error": f"Invalid faction: {faction}"},
                status=400,
            )

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON"},
                status=400,
            )

        # Handle single command or batch
        if "commands" in data:
            commands_data = data["commands"]
        else:
            commands_data = [data]

        # Parse and store commands
        accepted = []
        errors = []

        for cmd_data in commands_data:
            try:
                command_type_str = cmd_data.get("command_type", "")
                command_type = MCPCommandType(command_type_str)

                command = MCPCommand(
                    command_type=command_type,
                    ship_id=cmd_data.get("ship_id"),
                    parameters=cmd_data.get("parameters", {}),
                    timestamp=cmd_data.get("timestamp", 0.0),
                )

                self._state.add_command(faction, command)
                accepted.append({
                    "command_type": command_type_str,
                    "ship_id": cmd_data.get("ship_id"),
                })

            except (ValueError, KeyError) as e:
                errors.append({
                    "command_type": cmd_data.get("command_type"),
                    "error": str(e),
                })

        return web.json_response({
            "accepted": len(accepted),
            "commands": accepted,
            "errors": errors,
        })

    async def _handle_ready(self, request: web.Request) -> web.Response:
        """Handle ready signal from a faction."""
        faction = request.match_info["faction"]

        if faction not in ("alpha", "beta"):
            return web.json_response(
                {"error": f"Invalid faction: {faction}"},
                status=400,
            )

        # Signal ready
        self._state.signal_ready(faction)

        # Call callback if registered
        if self._on_ready_callback:
            try:
                await self._on_ready_callback(faction)
            except Exception as e:
                print(f"[MCPHttpServer] Error in ready callback: {e}")

        return web.json_response({
            "status": "ready",
            "faction": faction,
            "checkpoint": self._current_checkpoint,
        })


async def run_battle_with_http_server(
    battle_runner: Any,
    fleet_data: Dict[str, Any],
    host: str = "localhost",
    port: int = 8765,
) -> Any:
    """
    Run a fleet battle with HTTP server for MCP communication.

    Args:
        battle_runner: LLMBattleRunner instance
        fleet_data: Ship specifications
        host: Host for HTTP server
        port: Port for HTTP server

    Returns:
        BattleResult
    """
    from .mcp_controller import MCPController

    # Create and start HTTP server
    http_server = MCPHttpServer(host=host, port=port)

    # Register controllers if present
    if battle_runner.alpha_mcp:
        http_server.register_controller("alpha", battle_runner.alpha_mcp)
    if battle_runner.beta_mcp:
        http_server.register_controller("beta", battle_runner.beta_mcp)

    try:
        await http_server.start()

        # Run the battle with HTTP-based MCP communication
        result = await run_fleet_battle_with_http(
            battle_runner,
            fleet_data,
            http_server,
        )

        return result

    finally:
        await http_server.stop()


async def run_fleet_battle_with_http(
    runner: Any,
    fleet_data: Dict[str, Any],
    http_server: MCPHttpServer,
) -> Any:
    """
    Run fleet battle with HTTP-based MCP coordination.

    This is similar to run_fleet_battle_async but uses HTTP for MCP communication
    instead of in-process shared state.
    """
    from ..simulation import CombatSimulation, ManeuverType, Maneuver
    from .mcp_controller import apply_mcp_commands_to_simulation

    runner.setup_fleet_battle(fleet_data)

    # Get decision interval from fleet config
    decision_interval = runner.fleet_config.decision_interval_s if runner.fleet_config else 30.0

    # Skip personality selection for MCP-controlled fleets
    if runner.config.personality_selection:
        if runner.config.verbose:
            print("\n=== PERSONALITY SELECTION PHASE ===")

        # Let non-MCP Admirals choose personality
        if runner.alpha_admiral and not runner.alpha_mcp:
            if runner.config.verbose:
                print(f"\n[Admiral {runner.alpha_admiral.name}] Defining command personality...")
            try:
                personality = runner.alpha_admiral.select_personality(
                    num_ships=len(runner.alpha_ships),
                    verbose=False,
                )
                if runner.config.verbose:
                    desc = personality.get("personality_description", "")
                    if desc:
                        print(f"  {desc}")
            except Exception as e:
                if runner.config.verbose:
                    print(f"  [ERROR] Admiral personality selection failed: {e}")

        if runner.beta_admiral and not runner.beta_mcp:
            if runner.config.verbose:
                print(f"\n[Admiral {runner.beta_admiral.name}] Defining command personality...")
            try:
                personality = runner.beta_admiral.select_personality(
                    num_ships=len(runner.beta_ships),
                    verbose=False,
                )
                if runner.config.verbose:
                    desc = personality.get("personality_description", "")
                    if desc:
                        print(f"  {desc}")
            except Exception as e:
                if runner.config.verbose:
                    print(f"  [ERROR] Admiral personality selection failed: {e}")

    # Track time for Admiral pre-snapshots
    next_checkpoint_time = decision_interval

    # Advance chat turn at start
    if runner.mcp_chat:
        runner.mcp_chat.new_turn()

    # Get shared state for MCP coordination
    shared_state = get_mcp_state()
    shared_state.set_event_loop(asyncio.get_event_loop())

    # Determine which factions need MCP
    mcp_factions = []
    if runner.alpha_mcp:
        mcp_factions.append("alpha")
        shared_state.register_faction("alpha")
    if runner.beta_mcp:
        mcp_factions.append("beta")
        shared_state.register_faction("beta")

    while not runner._is_fleet_battle_over():
        # === SIMULATION PHASE ===
        steps = int(decision_interval)
        for step_i in range(steps):
            current_time = runner.simulation.current_time

            # Capture Admiral pre-snapshots at T-15s before checkpoint
            if current_time == next_checkpoint_time - runner.admiral_pre_snapshot_offset:
                runner._capture_admiral_pre_snapshots()

            runner.simulation.step()

            # Record sim frame if enabled
            if runner.recorder and runner.config.record_sim_trace:
                runner._record_sim_frame()

            if runner._is_fleet_battle_over():
                break

        if runner._is_fleet_battle_over():
            break

        # === CHECKPOINT ===
        runner.checkpoint_count += 1
        next_checkpoint_time = runner.simulation.current_time + decision_interval

        if runner.config.verbose:
            print(f"\n=== CHECKPOINT {runner.checkpoint_count} at T+{runner.simulation.current_time:.0f}s ===")
            runner._print_fleet_status()

        # Advance chat turn
        if runner.mcp_chat:
            runner.mcp_chat.new_turn()

        # Update battle status
        http_server.set_battle_status(
            "paused",
            checkpoint=runner.checkpoint_count,
            waiting_for=mcp_factions.copy(),
        )

        # === MCP/ADMIRAL DECISION PHASE ===
        admiral_orders = {}
        all_commands = {}

        # Update state and wait for MCP factions
        for faction in mcp_factions:
            controller = runner.alpha_mcp if faction == "alpha" else runner.beta_mcp
            captains = list(runner.alpha_captains.values()) if faction == "alpha" else list(runner.beta_captains.values())

            if controller:
                # Update state for MCP client to read
                controller.update_battle_state(runner.simulation, captains)

                if runner.config.verbose:
                    print(f"\n--- MCP COMMAND PHASE ({faction.title()}: {controller.name}) ---")
                    print(f"  State updated. Waiting for MCP client commands (no timeout)...")

        # Wait indefinitely for all MCP factions to signal ready
        for faction in mcp_factions:
            shared_state.clear_ready(faction)

        # Wait for all MCP factions (no timeout)
        while mcp_factions:
            for faction in list(mcp_factions):
                if shared_state.is_ready(faction):
                    mcp_factions.remove(faction)
                    http_server.set_battle_status(
                        "paused",
                        checkpoint=runner.checkpoint_count,
                        waiting_for=mcp_factions.copy(),
                    )

                    if runner.config.verbose:
                        print(f"  [{faction.title()}] Ready signal received")

                    # Process commands
                    commands = shared_state.get_pending_commands(faction)
                    if runner.config.verbose:
                        print(f"  Received {len(commands)} commands")

                    # Apply commands to simulation
                    results = apply_mcp_commands_to_simulation(
                        commands, runner.simulation, faction
                    )

                    if runner.config.verbose and results.get("applied"):
                        for cmd_result in results["applied"]:
                            print(f"    Applied: {cmd_result}")

            if mcp_factions:
                await asyncio.sleep(0.1)

        # Reset mcp_factions for next checkpoint
        mcp_factions = []
        if runner.alpha_mcp:
            mcp_factions.append("alpha")
        if runner.beta_mcp:
            mcp_factions.append("beta")

        # Handle non-MCP admirals
        if runner.alpha_admiral and not runner.alpha_mcp:
            active_alpha_captains = [
                c for c in runner.alpha_captains.values()
                if not getattr(runner.alpha_ships.get(c.ship_id), 'is_surrendered', False)
            ]
            alpha_decision = runner._get_admiral_decision(
                runner.alpha_admiral,
                active_alpha_captains,
                runner.beta_admiral,
            )
            for order in alpha_decision.fleet_orders:
                ship_id = runner._find_ship_id_by_name(order.target_ship_id, "alpha")
                if ship_id and ship_id in runner.alpha_captains:
                    if ship_id not in admiral_orders:
                        admiral_orders[ship_id] = []
                    admiral_orders[ship_id].append(order)

        if runner.beta_admiral and not runner.beta_mcp:
            active_beta_captains = [
                c for c in runner.beta_captains.values()
                if not getattr(runner.beta_ships.get(c.ship_id), 'is_surrendered', False)
            ]
            print(f"[DEBUG] Beta admiral decision: {len(active_beta_captains)} active captains")
            try:
                beta_decision = runner._get_admiral_decision(
                    runner.beta_admiral,
                    active_beta_captains,
                    runner.alpha_admiral,
                )
                print(f"[DEBUG] Beta admiral returned {len(beta_decision.fleet_orders)} fleet orders")
                for order in beta_decision.fleet_orders:
                    ship_id = runner._find_ship_id_by_name(order.target_ship_id, "beta")
                    print(f"[DEBUG] Beta order for {order.target_ship_id} -> ship_id={ship_id}: {order.order_text[:50]}...")
                    if ship_id and ship_id in runner.beta_captains:
                        if ship_id not in admiral_orders:
                            admiral_orders[ship_id] = []
                        admiral_orders[ship_id].append(order)
            except Exception as e:
                print(f"[ERROR] Beta admiral decision failed: {e}")
                import traceback
                traceback.print_exc()

        # Process captain decisions for non-MCP fleets
        # (MCP fleets have their commands applied directly via apply_mcp_commands_to_simulation)
        all_commands = {}

        # Process non-MCP alpha captains
        if not runner.alpha_mcp:
            if runner.config.verbose and runner.alpha_captains:
                print(f"\n--- CAPTAIN DECISIONS (Alpha) ---")
            for ship_id, captain in runner.alpha_captains.items():
                ship = runner.alpha_ships.get(ship_id)
                if ship and ship.is_destroyed:
                    continue

                # Clear previous context and give new orders
                captain.clear_admiral_context()
                if ship_id in admiral_orders:
                    orders = admiral_orders[ship_id]
                    directive = runner.alpha_admiral.last_directive if runner.alpha_admiral and hasattr(runner.alpha_admiral, 'last_directive') else None
                    captain.receive_admiral_orders(orders, directive)

                if runner.config.verbose:
                    print(f"  [{ship_id}] {captain.name} deciding...")

                # Get captain decision with discussion support
                commands = runner._get_captain_decision_with_discussion(
                    ship_id, captain, "alpha"
                )
                all_commands[ship_id] = commands

                if runner.config.verbose:
                    print(f"    -> {runner._get_ship_status_line(ship_id, commands)}")

        # Process non-MCP beta captains
        if not runner.beta_mcp:
            print(f"[DEBUG] Processing beta captains: {len(runner.beta_captains)} captains")
            if runner.config.verbose and runner.beta_captains:
                print(f"\n--- CAPTAIN DECISIONS (Beta) ---")
            for ship_id, captain in runner.beta_captains.items():
                ship = runner.beta_ships.get(ship_id)
                if ship and ship.is_destroyed:
                    print(f"[DEBUG] {ship_id} is destroyed, skipping")
                    continue

                # Clear previous context and give new orders
                captain.clear_admiral_context()
                orders_for_captain = admiral_orders.get(ship_id, [])
                print(f"[DEBUG] {ship_id} has {len(orders_for_captain)} admiral orders")
                if ship_id in admiral_orders:
                    orders = admiral_orders[ship_id]
                    directive = runner.beta_admiral.last_directive if runner.beta_admiral and hasattr(runner.beta_admiral, 'last_directive') else None
                    captain.receive_admiral_orders(orders, directive)

                if runner.config.verbose:
                    print(f"  [{ship_id}] {captain.name} deciding...")

                # Get captain decision with discussion support
                try:
                    print(f"[DEBUG] {ship_id} calling captain.decide...")
                    commands = runner._get_captain_decision_with_discussion(
                        ship_id, captain, "beta"
                    )
                    print(f"[DEBUG] {ship_id} captain returned {len(commands)} commands: {[type(c).__name__ for c in commands]}")
                    all_commands[ship_id] = commands
                except Exception as e:
                    print(f"[ERROR] {ship_id} captain decision failed: {e}")
                    import traceback
                    traceback.print_exc()
                    all_commands[ship_id] = []

                if runner.config.verbose:
                    print(f"    -> {runner._get_ship_status_line(ship_id, commands)}")

        # Handle immediate messaging
        runner._handle_immediate_messaging()

        # Check surrender/draw
        runner._check_fleet_surrender_draw()

        # Apply non-MCP captain commands to simulation via inject_command
        print(f"[DEBUG] Applying commands for {len(all_commands)} ships")
        for ship_id, commands in all_commands.items():
            print(f"[DEBUG] {ship_id}: {len(commands)} commands to apply")
            for cmd in commands:
                # Filter out discussion markers
                if isinstance(cmd, dict) and cmd.get('type') == 'discuss_with_admiral':
                    continue
                print(f"[DEBUG] {ship_id} injecting command: {type(cmd).__name__}")
                success = runner.simulation.inject_command(ship_id, cmd)
                print(f"[DEBUG] {ship_id} inject result: {success}")
                if runner.config.verbose and isinstance(cmd, dict) and cmd.get('type') == 'fire_at':
                    print(f"    [FIRE] {ship_id} {cmd.get('weapon_slot')} -> {'HIT' if success else 'FAILED'}")

        # Check limits
        if not runner.config.unlimited_mode:
            max_checkpoints = runner.fleet_config.max_checkpoints if runner.fleet_config and hasattr(runner.fleet_config, 'max_checkpoints') else runner.config.max_checkpoints
            if runner.checkpoint_count >= max_checkpoints:
                if runner.config.verbose:
                    print(f"\n=== CHECKPOINT LIMIT REACHED ===")
                break

    http_server.set_battle_status("ended", checkpoint=runner.checkpoint_count)
    return runner._evaluate_fleet_result()
