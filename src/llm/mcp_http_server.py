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
import dataclasses
import json
import math
import time
import uuid
from bisect import bisect_right
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
    from .mcp_draft import DraftManager

# A spectator counts as watching while it polled /live/recording this
# recently: the viewer polls every 2 s and backs off to 15 s between misses.
LIVE_CLIENT_ACTIVE_S = 20.0
# "ended" responses a viewer needs: the poll that carries the end, then the
# viewer's final catch-up fetch (finishLiveBattle in visualizer/src/main.js).
LIVE_ENDED_FETCHES = 2


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def _cors_middleware(request, handler):
        """
        Allow cross-origin reads so the Vite dev viewer (port 5173) can poll
        the live endpoints directly. The API is localhost-bound and read/
        command traffic is same-machine, so a wildcard is fine here.
        """
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            exc.headers["Access-Control-Allow-Origin"] = "*"
            raise
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
else:  # pragma: no cover - aiohttp is a hard dep of this module anyway
    _cors_middleware = None


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
        self._status_since = time.monotonic()

        # Live spectator wire. battle_id identifies this battle to the viewer:
        # a restarted server (or the next battle on the same port) serves a
        # new id, so the viewer resets instead of merging two battles.
        self.battle_id = uuid.uuid4().hex[:12]
        self.replay_url: Optional[str] = None       # saved replay the viewer can fall back to
        self.models: Optional[Dict[str, str]] = None  # faction -> commander model, for the HUD
        self.spend_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self.pending_calls: Any = None  # live_status.PendingCalls (else the recorder's)
        self._live_polls = 0           # /live/recording responses served
        self._live_ended_polls = 0     # ... of which carried status "ended"
        # Per spectator (the viewer's `client` id, else its remote address):
        # {"last": monotonic time of its last poll, "ended": "ended" responses}
        self._live_clients: Dict[str, Dict[str, float]] = {}

        # Controllers for building state
        self._controllers: Dict[str, 'MCPController'] = {}

        # Pre-battle draft phase (None outside draft-enabled battles)
        self._draft_manager: Optional['DraftManager'] = None

        # Battle runner backing the /live endpoints (set before the battle
        # starts so spectators can watch the draft phase too). NOT self._runner,
        # which is the aiohttp AppRunner.
        self._battle_runner: Any = None

        # Callbacks
        self._on_ready_callback: Optional[Callable[[str], Awaitable[None]]] = None

        # Server instances
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    def register_controller(self, faction: str, controller: 'MCPController') -> None:
        """Register an MCP controller for a faction."""
        self._controllers[faction] = controller

    def set_draft_manager(self, manager: 'DraftManager') -> None:
        """Attach the draft phase manager (routes /draft and draft-time ready)."""
        self._draft_manager = manager

    def set_live_source(self, runner: Any) -> None:
        """Attach the battle runner backing the /live spectator endpoints."""
        self._battle_runner = runner

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
        if status != self._battle_status or checkpoint != self._current_checkpoint:
            self._status_since = time.monotonic()
        self._battle_status = status
        self._current_checkpoint = checkpoint
        self._waiting_for = waiting_for or []

    async def linger_for_spectators(self, timeout_s: float = 10.0) -> None:
        """
        Keep serving after the battle ended until every spectator still
        watching has fetched the end - its "ended" poll and the follow-up
        final fetch - instead of finding a dead socket on its next poll.
        Spectators are told apart by the viewer's `client` query parameter
        (viewers without one share an entry per remote address); one counts
        as watching while it polled within LIVE_CLIENT_ACTIVE_S. Returns at
        once when nobody is watching, at timeout_s regardless.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            now = time.monotonic()
            waiting = [c for c in self._live_clients.values()
                       if now - c["last"] <= LIVE_CLIENT_ACTIVE_S
                       and c["ended"] < LIVE_ENDED_FETCHES]
            if not waiting or now >= deadline:
                return
            await asyncio.sleep(0.1)

    def _note_live_poll(self, request: 'web.Request', ended: bool) -> None:
        """Book one /live/recording response against its spectator."""
        self._live_polls += 1
        if ended:
            self._live_ended_polls += 1
        key = request.query.get("client", "")[:64] or f"remote:{request.remote}"
        now = time.monotonic()
        client = self._live_clients.get(key)
        if client is None:
            if len(self._live_clients) >= 256:   # forget long-gone viewers
                for gone in [k for k, c in self._live_clients.items()
                             if now - c["last"] > LIVE_CLIENT_ACTIVE_S]:
                    del self._live_clients[gone]
            client = self._live_clients[key] = {"last": now, "ended": 0}
        client["last"] = now
        if ended:
            client["ended"] += 1

    def build_app(self) -> 'web.Application':
        """Build the aiohttp application (separate from start() for tests)."""
        self._app = web.Application(middlewares=[_cors_middleware])
        self._setup_routes()
        return self._app

    async def start(self) -> None:
        """Start the HTTP server."""
        self.build_app()

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
        # Draft phase (only meaningful while a DraftManager is attached)
        self._app.router.add_get("/draft/{faction}", self._handle_get_draft)
        self._app.router.add_post("/draft/{faction}/select", self._handle_draft_select)
        self._app.router.add_post("/draft/{faction}/formation", self._handle_draft_formation)
        # Live spectator endpoints (the web viewer polls these)
        self._app.router.add_get("/live/recording", self._handle_live_recording)
        self._app.router.add_get("/live/predictions", self._handle_live_predictions)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok"})

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Get battle status."""
        drafting = self._draft_manager is not None and self._draft_manager.active
        return web.json_response({
            "status": self._battle_status,
            "phase": "draft" if drafting else (
                "ended" if self._battle_status == "ended" else "battle"),
            "checkpoint": self._current_checkpoint,
            "waiting_for": (self._draft_manager.waiting_for() if drafting
                            else self._waiting_for),
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

        # During the draft phase, ready commits the faction's draft instead
        # of advancing a (not yet existing) checkpoint.
        if self._draft_manager is not None and self._draft_manager.active:
            ok, error = self._draft_manager.commit(faction)
            if not ok:
                return web.json_response(
                    {"status": "error", "error": error},
                    status=409,
                )
            return web.json_response({
                "status": "draft_committed",
                "faction": faction,
                "waiting_for": self._draft_manager.waiting_for(),
            })

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

    # === Draft phase endpoints ===

    def _draft_request_faction(self, request: web.Request):
        """Common validation for /draft routes: (faction, error_response)."""
        faction = request.match_info["faction"]
        if faction not in ("alpha", "beta"):
            return None, web.json_response(
                {"error": f"Invalid faction: {faction}"}, status=400)
        if self._draft_manager is None:
            return None, web.json_response(
                {"error": "This battle has no draft phase "
                          "(no 'draft' block in the fleet config)."},
                status=404)
        return faction, None

    async def _handle_get_draft(self, request: web.Request) -> web.Response:
        """Draft state for a faction: budget, catalog, current picks."""
        faction, err = self._draft_request_faction(request)
        if err:
            return err
        return web.json_response(self._draft_manager.state_dict(faction))

    async def _handle_draft_select(self, request: web.Request) -> web.Response:
        """Apply a fleet selection; validation errors come back as 409."""
        faction, err = self._draft_request_faction(request)
        if err:
            return err
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        result, error = self._draft_manager.select(
            faction, data.get("ships"), data.get("rationale", ""))
        if error:
            return web.json_response(
                {"status": "error", "error": error}, status=409)
        return web.json_response({"status": "ok", **result})

    async def _handle_draft_formation(self, request: web.Request) -> web.Response:
        """Apply formation placements; coercions are reported as notes."""
        faction, err = self._draft_request_faction(request)
        if err:
            return err
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        result, error = self._draft_manager.formation(
            faction,
            data.get("placements"),
            data.get("formation_name", ""),
            data.get("rationale", ""),
        )
        if error:
            return web.json_response(
                {"status": "error", "error": error}, status=409)
        return web.json_response({"status": "ok", **result})

    # === Live spectator endpoints ===

    def _pending_registry(self) -> Any:
        """The PendingCalls registry: set on the server, else the recorder's."""
        if self.pending_calls is not None:
            return self.pending_calls
        recorder = getattr(self._battle_runner, "recorder", None) if self._battle_runner else None
        return getattr(recorder, "pending_calls", None) if recorder is not None else None

    def _live_models(self) -> Dict[str, Optional[str]]:
        """Commander per faction for the HUD: explicit, else MCP / admiral model."""
        if self.models:
            return dict(self.models)
        runner = self._battle_runner
        models: Dict[str, Optional[str]] = {}
        for faction in ("alpha", "beta"):
            if runner is not None and getattr(runner, f"{faction}_mcp", None):
                models[faction] = "MCP"
                continue
            admiral = getattr(runner, f"{faction}_admiral", None) if runner is not None else None
            models[faction] = getattr(admiral, "model", None) if admiral is not None else None
        return models

    def _live_block(self) -> Dict[str, Any]:
        """
        Battle-flow metadata block served with every /live response.

        Besides the flow fields (phase/status/checkpoint/sim clock) it
        carries what a spectator needs to read a deliberation: battle_id
        (identity across server restarts), status_age_s (wall seconds in the
        current status, e.g. how long the commanders have been deciding),
        pending_calls (commander model calls in flight, with their age),
        spend/models when the runner provides them, and replay_url, the
        saved recording to fall back to when the server goes away.
        """
        runner = self._battle_runner
        drafting = self._draft_manager is not None and self._draft_manager.active
        phase = "draft" if drafting else (
            "ended" if self._battle_status == "ended" else "battle")
        sim = getattr(runner, "simulation", None) if runner else None
        sim_time = float(sim.current_time) if sim else 0.0
        fleet_config = getattr(runner, "fleet_config", None) if runner else None
        interval = float(getattr(fleet_config, "decision_interval_s", 30.0) or 30.0)
        next_cp = (math.floor(sim_time / interval + 1e-9) + 1) * interval
        registry = self._pending_registry()
        pending = registry.snapshot() if registry is not None else []
        waiting = (self._draft_manager.waiting_for() if drafting
                   else list(self._waiting_for))
        if not drafting and self._battle_status == "paused":
            # MCP factions the server waits on, plus sides with model calls
            # in flight (LLM admirals and captains)
            waiting = sorted(set(waiting) | {c["faction"] for c in pending if c.get("faction")})
        spend = None
        if self.spend_provider is not None:
            try:
                spend = self.spend_provider()
            except Exception:  # noqa: BLE001 - spectator data must never break a poll
                spend = None
        recorder = getattr(runner, "recorder", None) if runner else None
        rec = getattr(recorder, "recording", None) if recorder else None
        result = None
        if self._battle_status == "ended" and rec is not None:
            result = {"winner": getattr(rec, "winner", None),
                      "reason": getattr(rec, "result_reason", None)}
        return {
            "phase": phase,
            "status": self._battle_status,
            "checkpoint": self._current_checkpoint,
            "decision_interval_s": interval,
            "sim_time_s": sim_time,
            "next_checkpoint_t": next_cp,
            "waiting_for": waiting,
            "draft": (self._draft_manager.live_summary()
                      if self._draft_manager else None),
            "battle_id": self.battle_id,
            "status_age_s": round(time.monotonic() - self._status_since, 1),
            "pending_calls": pending,
            "models": self._live_models(),
            "spend": spend,
            "replay_url": self.replay_url,
            "result": result,
        }

    async def _handle_live_recording(self, request: web.Request) -> web.Response:
        """
        The recording-so-far, shaped exactly like a saved BattleRecording so
        the web viewer's normal loader can parse it. `?since_t=` narrows
        sim_trace/events to entries after that sim time (metadata always
        included). `recording` is null until the battle is set up (draft
        phase, or recording/sim-trace disabled in the config). `?client=`
        names the spectator, so the end-of-battle linger waits for each one.
        """
        since_arg = request.query.get("since_t")
        try:
            since = float(since_arg) if since_arg is not None else None
        except ValueError:
            return web.json_response(
                {"error": f"Bad since_t: {since_arg!r}"}, status=400)

        try:
            since_seq = int(request.query['since_seq']) if 'since_seq' in request.query else None
            if since_seq is not None and since_seq < 0:
                raise ValueError
            if since is not None and not math.isfinite(since):
                raise ValueError
        except ValueError:
            return web.json_response({'error': 'Invalid recording cursor'}, status=400)

        recording = None
        runner = self._battle_runner
        recorder = getattr(runner, "recorder", None) if runner else None
        rec = getattr(recorder, "recording", None) if recorder else None
        if rec is not None and rec.is_fleet_battle:
            frames = rec.sim_trace
            events = [e.to_dict() for e in recorder.events]
            if since is not None:
                start = bisect_right(frames, since, key=lambda f: f["t"])
                frames = frames[start:]
                if since_seq is None:
                    events = [e for e in events if e['timestamp'] > since]
            if since_seq is not None:
                events = [e for e in events if e.get('sequence', 0) > since_seq]
            # Shallow field copy: asdict() would deep-copy the whole trace
            # on every poll.
            recording = {
                f.name: getattr(rec, f.name)
                for f in dataclasses.fields(rec)
                if f.name not in ("sim_trace", "events")
            }
            recording["sim_trace"] = frames
            recording["events"] = events
            recording['assets'] = dict(rec.assets)
            if since_seq is not None:
                refs = set()
                for event in events:
                    refs.update(event['data'].get('message_refs', []))
                    if event['data'].get('tools_ref'):
                        refs.add(event['data']['tools_ref'])
                recording['assets'] = {key: rec.assets[key] for key in refs if key in rec.assets}
        live = self._live_block()
        self._note_live_poll(request, live["status"] == "ended")
        return web.json_response({
            "live": live,
            "recording": recording,
        })

    async def _handle_live_predictions(self, request: web.Request) -> web.Response:
        """
        Predicted path of every live ship to the next checkpoint, from the
        last recorded frame. Positions in metres, matching frame `pos`.

        Serves the frame's recorded `plan` (movement-only look-ahead that
        follows the ship's own guidance and evasion, with `sigma` = 1-sigma
        dispersion of the checkpoint position, see
        src/trajectory_prediction.py). Frames without a plan fall back to
        constant-acceleration extrapolation (a derived from the last two
        frames, sigma 0), sampled every 2 s of sim time.

        `stale` (per ship, and top level when any ship's is) is true while
        the battle sits at a checkpoint whose orders are not applied yet
        (commanders deliberating): the path flies out the expiring orders.
        It clears once the orders land and the frame is re-predicted.
        """
        empty = {"t": 0.0, "t_checkpoint": 0.0, "ships": {}, "stale": False}
        runner = self._battle_runner
        recorder = getattr(runner, "recorder", None) if runner else None
        rec = getattr(recorder, "recording", None) if recorder else None
        drafting = self._draft_manager is not None and self._draft_manager.active
        if rec is None or not rec.sim_trace or drafting:
            return web.json_response(empty)

        frames = rec.sim_trace
        last = frames[-1]
        prev = frames[-2] if len(frames) >= 2 else None
        t1 = float(last["t"])
        fleet_config = getattr(runner, "fleet_config", None)
        interval = float(getattr(fleet_config, "decision_interval_s", 30.0) or 30.0)
        t_cp = (math.floor(t1 / interval + 1e-9) + 1) * interval
        horizon = t_cp - t1

        ships: Dict[str, Any] = {}
        for ship_id, state in last.get("ships", {}).items():
            if state.get("destroyed") or state.get("dying"):
                continue
            plan = state.get("plan")
            if plan and plan.get("path"):
                t_cp = float(plan.get("t_cp", t_cp))
                ships[ship_id] = {
                    "path": plan["path"],
                    "checkpoint_pos": plan["path"][-1],
                    "sigma": plan.get("sigma", 0.0),
                    "dt": plan.get("dt"),
                    "stale": bool(plan.get("stale", False)),
                }
                continue
            p1 = state["pos"]
            v1 = state["vel"]
            accel = [0.0, 0.0, 0.0]
            if prev is not None:
                s0 = prev.get("ships", {}).get(ship_id)
                if s0 and not s0.get("destroyed"):
                    dt0 = t1 - float(prev["t"])
                    if dt0 > 0:
                        accel = [(v1[i] - s0["vel"][i]) / dt0 for i in range(3)]

            def _at(dt: float) -> List[float]:
                return [p1[i] + v1[i] * dt + 0.5 * accel[i] * dt * dt
                        for i in range(3)]

            path = []
            dt = 0.0
            while dt < horizon - 1e-9:
                path.append(_at(dt))
                dt += 2.0
            path.append(_at(horizon))
            ships[ship_id] = {"path": path, "checkpoint_pos": path[-1], "sigma": 0.0,
                              "stale": False}

        return web.json_response({
            "t": t1,
            "t_checkpoint": t_cp,
            "ships": ships,
            "stale": any(s["stale"] for s in ships.values()),
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
    # Create and start HTTP server. Controllers are registered inside
    # run_fleet_battle_with_http once setup_fleet_battle has created them -
    # registering battle_runner.alpha_mcp here always stored None.
    http_server = MCPHttpServer(host=host, port=port)
    http_server.set_live_source(battle_runner)

    try:
        await http_server.start()

        # Optional pre-battle draft phase: fleets are bought and placed
        # before the simulation exists.
        draft_cfg = getattr(battle_runner.fleet_config, "draft", None)
        if draft_cfg is not None and draft_cfg.enabled:
            await run_mcp_draft_phase(battle_runner, fleet_data, http_server)

        # Run the battle with HTTP-based MCP communication
        result = await run_fleet_battle_with_http(
            battle_runner,
            fleet_data,
            http_server,
        )

        return result

    finally:
        # A polling spectator still gets the final "ended" block
        await http_server.linger_for_spectators()
        await http_server.stop()


async def run_mcp_draft_phase(
    runner: Any,
    fleet_data: Dict[str, Any],
    http_server: MCPHttpServer,
) -> None:
    """
    Run the pre-battle draft: MCP factions draft over the HTTP API
    (get_draft_state / select_fleet / set_formation / ready), non-MCP
    factions draft via their admiral LLM or the deterministic auto-draft.
    Blocks until every faction has committed, then replaces the fleet
    definitions on runner.fleet_config with the drafted fleets.
    """
    from .fleet_draft import auto_draft, draft_to_fleet_definition, run_admiral_draft
    from .mcp_draft import DraftManager

    cfg = runner.fleet_config
    draft_cfg = cfg.draft
    fleets = {"alpha": cfg.alpha_fleet, "beta": cfg.beta_fleet}
    mcp_factions = {f for f, fleet in fleets.items()
                    if fleet.mcp and fleet.mcp.enabled}

    manager = DraftManager(
        fleet_data=fleet_data,
        budget=draft_cfg.budget,
        max_ships=draft_cfg.max_ships,
        initial_distance_km=cfg.initial_distance_km,
        mcp_factions=mcp_factions,
    )
    http_server.set_draft_manager(manager)
    http_server.set_battle_status("drafting", checkpoint=0,
                                  waiting_for=sorted(mcp_factions))

    print(f"\n=== DRAFT PHASE === "
          f"{draft_cfg.budget} points, max {draft_cfg.max_ships} hulls")
    for faction in sorted(mcp_factions):
        print(f"  [{faction}] waiting for MCP draft "
              f"(get_draft_state -> select_fleet -> set_formation -> ready)")

    loop = asyncio.get_event_loop()

    async def draft_non_mcp(faction: str) -> None:
        fleet = fleets[faction]
        if fleet.admiral is not None and fleet.admiral.enabled:
            admiral = fleet.admiral
            # run_admiral_draft is synchronous LLM I/O - keep the event loop
            # (and the MCP faction's draft endpoints) responsive meanwhile.
            draft = await loop.run_in_executor(None, lambda: run_admiral_draft(
                runner.client,
                admiral.model,
                admiral.name or f"Admiral {faction.title()}",
                faction,
                fleet_data,
                budget=draft_cfg.budget,
                max_ships=draft_cfg.max_ships,
                initial_distance_km=cfg.initial_distance_km,
                verbose=runner.config.verbose,
                seed=runner.config.seed,
            ))
        else:
            draft = auto_draft(faction, draft_cfg.budget, draft_cfg.max_ships,
                               seed=runner.config.seed)
            roster = ", ".join(s.ship_name for s in draft.ships)
            print(f"[DRAFT {faction}] auto-draft "
                  f"({draft.points_spent}/{draft_cfg.budget} pts): {roster}")
        manager.set_full_draft(faction, draft)

    non_mcp_tasks = [asyncio.ensure_future(draft_non_mcp(f))
                     for f in fleets if f not in mcp_factions]

    waited_s = 0.0
    while not manager.all_committed():
        await asyncio.sleep(0.1)
        waited_s += 0.1
        if waited_s >= 30.0:
            waited_s = 0.0
            print(f"  [DRAFT] Still waiting for: "
                  f"{', '.join(manager.waiting_for())}")
        http_server.set_battle_status("drafting", checkpoint=0,
                                      waiting_for=manager.waiting_for())
    if non_mcp_tasks:
        await asyncio.gather(*non_mcp_tasks)
    manager.finalize()

    # Materialize the drafts into fleet definitions (positions in km)
    for faction, fleet in fleets.items():
        draft = manager.slot(faction).draft
        is_mcp = faction in mcp_factions
        definition = draft_to_fleet_definition(
            draft,
            cfg.initial_distance_km,
            captain_model="mcp" if is_mcp else draft_cfg.captain_model,
            admiral_config=None if is_mcp else fleet.admiral,
            mcp_config=fleet.mcp if is_mcp else None,
        )
        if faction == "alpha":
            cfg.alpha_fleet = definition
        else:
            cfg.beta_fleet = definition
        roster = ", ".join(s.ship_name for s in definition.ships)
        commander = ("MCP" if is_mcp
                     else (fleet.admiral.name if fleet.admiral else "auto"))
        print(f"[DRAFT {faction}] committed ({commander}, "
              f"{draft.points_spent}/{draft_cfg.budget} pts, "
              f"formation '{draft.formation_name}'): {roster}")


def apply_mcp_fleet_control(runner: Any) -> bool:
    """
    Apply MCP fleet-level surrender / mutual-draw decisions to the battle.

    Mirrors the block in LLMBattleRunner.run_fleet_battle_async: the shared
    ``_check_fleet_surrender_draw()`` only looks at captains and LLM admirals,
    so MCP controllers' ``has_surrendered`` / ``has_proposed_draw`` /
    ``has_accepted_draw`` flags must be consumed explicitly.

    Returns:
        True if the battle should stop now (a mutual draw was agreed).
    """
    verbose = getattr(runner.config, "verbose", False)

    for faction, controller, ships in (
        ("alpha", runner.alpha_mcp, runner.alpha_ships),
        ("beta", runner.beta_mcp, runner.beta_ships),
    ):
        if controller and controller.has_surrendered:
            for ship_id in ships:
                ship = runner.simulation.get_ship(ship_id)
                if ship:
                    ship.is_surrendered = True
            if verbose:
                print(f"  [SURRENDER] {controller.name} surrenders")

    if runner.alpha_mcp and runner.beta_mcp:
        mutual = (
            (runner.alpha_mcp.has_proposed_draw and runner.beta_mcp.has_accepted_draw)
            or (runner.beta_mcp.has_proposed_draw and runner.alpha_mcp.has_accepted_draw)
        )
        if mutual:
            if verbose:
                print("  [DRAW ACCEPTED] Mutual draw agreed")
            return True

    return False


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
    _attach_live_status(runner, http_server)

    # Controllers exist only after setup - register them for /status introspection.
    if runner.alpha_mcp:
        http_server.register_controller("alpha", runner.alpha_mcp)
    if runner.beta_mcp:
        http_server.register_controller("beta", runner.beta_mcp)

    def dbg(msg: str) -> None:
        """Debug trace, gated on verbose so --quiet is actually quiet."""
        if runner.config.verbose:
            print(msg)

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
        http_server.set_battle_status(
            "running", checkpoint=runner.checkpoint_count, waiting_for=[])
        for _ in runner._advance_to_checkpoint(next_checkpoint_time, fleet_mode=True):

            # Record sim frame if enabled
            if runner.recorder and runner.config.record_sim_trace:
                runner._record_sim_frame()

            # Yield so live-viewer polls get served between sim steps
            await asyncio.sleep(0)

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
            enemy_controller = runner.beta_mcp if faction == "alpha" else runner.alpha_mcp
            enemy_admiral = runner.beta_admiral if faction == "alpha" else runner.alpha_admiral
            captains = list(runner.alpha_captains.values()) if faction == "alpha" else list(runner.beta_captains.values())

            if controller:
                # Publish turn number / battle liveness / enemy draw proposal so
                # the MCP client can actually see them (they used to be hardcoded).
                enemy_side = enemy_controller or enemy_admiral
                controller.set_battle_progress(
                    checkpoint_number=runner.checkpoint_count,
                    is_battle_active=not runner._is_fleet_battle_over(),
                    enemy_proposed_draw=bool(
                        getattr(enemy_side, "has_proposed_draw", False)
                    ),
                )

                # Update state for MCP client to read
                controller.update_battle_state(runner.simulation, captains)

                if runner.config.verbose:
                    print(f"\n--- MCP COMMAND PHASE ({faction.title()}: {controller.name}) ---")
                    print(f"  State updated. Waiting for MCP client commands (no timeout)...")

        # Wait indefinitely for all MCP factions to signal ready
        for faction in mcp_factions:
            shared_state.clear_ready(faction)

        # Wait for all MCP factions (no timeout, but with periodic diagnostics so
        # a side that forgets ready() does not hang the battle silently)
        waited_s = 0.0
        heartbeat_s = 30.0
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

                    # Fleet-level control commands (message / draw / surrender)
                    # are controller state, not simulation state - they must be
                    # applied here or the tools are silent no-ops on this path.
                    controller = runner.alpha_mcp if faction == "alpha" else runner.beta_mcp
                    if controller:
                        control_applied = controller.process_control_commands(
                            commands, runner.simulation.current_time
                        )
                        if runner.config.verbose:
                            for entry in control_applied:
                                print(f"    Applied: {entry}")

                    # Apply commands to simulation
                    results = apply_mcp_commands_to_simulation(
                        commands, runner.simulation, faction
                    )

                    if runner.config.verbose and results.get("applied"):
                        for cmd_result in results["applied"]:
                            print(f"    Applied: {cmd_result}")

                    # Errors used to be dropped on the floor: a bad ship id or
                    # maneuver name vanished with no signal to anyone.
                    for cmd_error in results.get("errors", []):
                        print(f"    [MCP ERROR] {cmd_error}")

            if mcp_factions:
                await asyncio.sleep(0.1)
                waited_s += 0.1
                if waited_s >= heartbeat_s:
                    waited_s = 0.0
                    print(
                        f"  [MCP] Still waiting for ready() from: {', '.join(mcp_factions)} "
                        f"(checkpoint {runner.checkpoint_count})"
                    )

        # === MESSAGE BRIDGE: MCP <-> LLM Admiral (and MCP <-> MCP) ===
        # Shared with run_fleet_battle_async (it used to be two copies, and
        # this fork once lost it): delivers and records the diplomacy.
        runner._bridge_mcp_messages()

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
                and not getattr(runner.alpha_ships.get(c.ship_id), 'is_dying', False)
            ]
            alpha_decision = runner._get_admiral_decision(
                runner.alpha_admiral,
                active_alpha_captains,
                runner.beta_admiral,
            )
            runner._record_admiral_intent(runner.alpha_admiral, alpha_decision, 'alpha')
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
                and not getattr(runner.beta_ships.get(c.ship_id), 'is_dying', False)
            ]
            dbg(f"[DEBUG] Beta admiral decision: {len(active_beta_captains)} active captains")
            try:
                beta_decision = runner._get_admiral_decision(
                    runner.beta_admiral,
                    active_beta_captains,
                    runner.alpha_admiral,
                )
                dbg(f"[DEBUG] Beta admiral returned {len(beta_decision.fleet_orders)} fleet orders")
                runner._record_admiral_intent(runner.beta_admiral, beta_decision, 'beta')
                for order in beta_decision.fleet_orders:
                    ship_id = runner._find_ship_id_by_name(order.target_ship_id, "beta")
                    dbg(f"[DEBUG] Beta order for {order.target_ship_id} -> ship_id={ship_id}: {order.order_text[:50]}...")
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

        # Fleet comms: one delivery round for every captain about to decide
        # (the HTTP loop does not go through runner._run_captain_decisions).
        deciding = [
            (ship_id, captain, faction)
            for faction, captains, ships, mcp in (
                ("alpha", runner.alpha_captains, runner.alpha_ships, runner.alpha_mcp),
                ("beta", runner.beta_captains, runner.beta_ships, runner.beta_mcp))
            if not mcp
            for ship_id, captain in captains.items()
            if not ((ship := ships.get(ship_id)) and (
                ship.is_destroyed or getattr(ship, 'is_dying', False)
                or getattr(ship, 'is_surrendered', False)))
        ]
        runner._deliver_fleet_messages(deciding)

        # Process non-MCP alpha captains
        if not runner.alpha_mcp:
            if runner.config.verbose and runner.alpha_captains:
                print(f"\n--- CAPTAIN DECISIONS (Alpha) ---")
            for ship_id, captain in runner.alpha_captains.items():
                ship = runner.alpha_ships.get(ship_id)
                # Surrendered ships must be skipped too: asking a surrendered
                # captain for orders (and paying for the LLM call) is exactly
                # what run_fleet_battle_async avoids.
                if ship and (ship.is_destroyed or getattr(ship, 'is_dying', False)
                             or getattr(ship, 'is_surrendered', False)):
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
                runner._finish_captain_turn(ship_id, captain, "alpha", commands,
                                            admiral_orders.get(ship_id))

                if runner.config.verbose:
                    print(f"    -> {runner._get_ship_status_line(ship_id, commands)}")

        # Process non-MCP beta captains
        if not runner.beta_mcp:
            dbg(f"[DEBUG] Processing beta captains: {len(runner.beta_captains)} captains")
            if runner.config.verbose and runner.beta_captains:
                print(f"\n--- CAPTAIN DECISIONS (Beta) ---")
            for ship_id, captain in runner.beta_captains.items():
                ship = runner.beta_ships.get(ship_id)
                if ship and (ship.is_destroyed or getattr(ship, 'is_dying', False)
                             or getattr(ship, 'is_surrendered', False)):
                    dbg(f"[DEBUG] {ship_id} is destroyed/dying/surrendered, skipping")
                    continue

                # Clear previous context and give new orders
                captain.clear_admiral_context()
                orders_for_captain = admiral_orders.get(ship_id, [])
                dbg(f"[DEBUG] {ship_id} has {len(orders_for_captain)} admiral orders")
                if ship_id in admiral_orders:
                    orders = admiral_orders[ship_id]
                    directive = runner.beta_admiral.last_directive if runner.beta_admiral and hasattr(runner.beta_admiral, 'last_directive') else None
                    captain.receive_admiral_orders(orders, directive)

                if runner.config.verbose:
                    print(f"  [{ship_id}] {captain.name} deciding...")

                # Get captain decision with discussion support
                try:
                    dbg(f"[DEBUG] {ship_id} calling captain.decide...")
                    commands = runner._get_captain_decision_with_discussion(
                        ship_id, captain, "beta"
                    )
                    dbg(f"[DEBUG] {ship_id} captain returned {len(commands)} commands: {[type(c).__name__ for c in commands]}")
                    all_commands[ship_id] = commands
                    runner._finish_captain_turn(ship_id, captain, "beta", commands,
                                                admiral_orders.get(ship_id))
                except Exception as e:
                    print(f"[ERROR] {ship_id} captain decision failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Reset `commands` too - the status line below would
                    # otherwise report the *previous* ship's command list.
                    commands = []
                    all_commands[ship_id] = []

                if runner.config.verbose:
                    print(f"    -> {runner._get_ship_status_line(ship_id, commands)}")

        # === MCP SURRENDER / MUTUAL DRAW ===
        # _check_fleet_surrender_draw() only inspects captains and LLM admirals,
        # so without this block surrender_fleet / propose_ + accept_fleet_draw
        # had no effect at all on the HTTP path.
        if apply_mcp_fleet_control(runner):
            break

        # Check surrender/draw
        runner._check_fleet_surrender_draw()

        if runner._is_fleet_battle_over():
            break

        # Apply non-MCP captain commands to simulation via inject_command
        dbg(f"[DEBUG] Applying commands for {len(all_commands)} ships")
        for ship_id, commands in all_commands.items():
            dbg(f"[DEBUG] {ship_id}: {len(commands)} commands to apply")
            for cmd in commands:
                # Filter out discussion markers
                if isinstance(cmd, dict) and cmd.get('type') == 'discuss_with_admiral':
                    continue
                dbg(f"[DEBUG] {ship_id} injecting command: {type(cmd).__name__}")
                success = runner.simulation.inject_command(ship_id, cmd)
                dbg(f"[DEBUG] {ship_id} inject result: {success}")
                if runner.config.verbose and isinstance(cmd, dict) and cmd.get('type') == 'fire_at':
                    print(f"    [FIRE] {ship_id} {cmd.get('weapon_slot')} -> {'HIT' if success else 'FAILED'}")

        # Log decision (the async loop does this; the fork had dropped it, so
        # runner.decision_log stayed empty for every HTTP battle)
        runner._log_fleet_decision(all_commands)

        # Check limits
        if not runner.config.unlimited_mode:
            max_checkpoints = runner.fleet_config.max_checkpoints if runner.fleet_config and hasattr(runner.fleet_config, 'max_checkpoints') else runner.config.max_checkpoints
            if runner.checkpoint_count >= max_checkpoints:
                if runner.config.verbose:
                    print(f"\n=== CHECKPOINT LIMIT REACHED ===")
                break

    # Verdict first: a spectator polling the moment the status flips to
    # "ended" must find the winner/reason already in the recording.
    result = runner._evaluate_fleet_result()
    http_server.set_battle_status("ended", checkpoint=runner.checkpoint_count)
    return result


def _attach_live_status(runner: Any, http_server: Any) -> None:
    """
    Spectator bookkeeping once the battle is set up: in-flight model calls
    (PendingCalls on the recorder, read by AuditedClient) and the battle_id
    in the recording's provenance so a saved replay is identifiable.
    attach_audit() replaces provenance during setup, hence after it.
    """
    from .live_status import PendingCalls
    recorder = getattr(runner, "recorder", None)
    if recorder is None or not hasattr(recorder, "recording"):
        return
    if getattr(recorder, "pending_calls", None) is None:
        recorder.pending_calls = PendingCalls()
    battle_id = getattr(http_server, "battle_id", None)
    if isinstance(battle_id, str):
        recorder.recording.provenance["battle_id"] = battle_id
