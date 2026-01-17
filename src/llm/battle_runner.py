"""
LLM Battle Runner - Orchestrates battles between LLM-controlled captains.

Handles simulation setup, checkpoint timing, and victory evaluation.
Supports both legacy 1v1 battles and multi-ship fleet battles with Admirals.
Also supports MCP-controlled fleets for external control (e.g., Claude Code).
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .client import CaptainClient
from .captain import LLMCaptain, LLMCaptainConfig
from .communication import CommunicationChannel, FleetCommunicationChannel, MessageType
from .victory import VictoryEvaluator, BattleOutcome
from .prompts import CaptainPersonality
from .battle_recorder import BattleRecorder, create_battle_filename
from .fleet_config import BattleFleetConfig, FleetDefinition, ShipConfig, AdmiralConfig, MCPConfig
from .admiral import LLMAdmiral, AdmiralOrder
from .mcp_controller import MCPController, MCPControllerConfig, apply_mcp_commands_to_simulation
from .mcp_chat import AdmiralChat


@dataclass
class BattleConfig:
    """Configuration for an LLM battle."""
    # Scenario
    initial_distance_km: float = 500.0  # Start closer for faster engagement
    initial_offset_km: float = 1.0  # Y-axis offset
    time_limit_s: float = 1200.0  # 20 minutes
    decision_interval_s: float = 30.0
    max_checkpoints: int = 40  # More passes

    # Unlimited mode - fight until destruction, surrender, or mutual draw
    # When enabled, time_limit_s and max_checkpoints are ignored
    unlimited_mode: bool = False

    # Ship types
    alpha_ship_type: str = "destroyer"
    beta_ship_type: str = "destroyer"

    # Verbose output
    verbose: bool = True

    # Personality selection - let LLMs choose their own personality before battle
    personality_selection: bool = True

    # Recording
    record_battle: bool = True
    recording_dir: str = "data/recordings"

    # Detailed sim trace - records every step (position, velocity for all objects)
    # WARNING: Generates large files (~1MB per 10 minutes of battle)
    record_sim_trace: bool = False

    # Fleet configuration (for multi-ship battles with Admirals)
    # If provided, overrides alpha/beta ship types and enables fleet mode
    fleet_config_path: Optional[str] = None


@dataclass
class BattleResult:
    """Result of an LLM battle."""
    outcome: BattleOutcome
    winner: Optional[str]
    reason: str
    duration_s: float
    checkpoints_used: int

    # Per-side stats (legacy 1v1 mode)
    alpha_stats: Dict[str, Any]
    beta_stats: Dict[str, Any]

    # Logs
    decision_log: List[Dict[str, Any]]
    messages: List[str]

    # Fields with defaults must come after fields without defaults
    # Recording file path (if recorded)
    recording_file: Optional[str] = None

    # Fleet stats (multi-ship mode) - per-ship stats by ship_id
    alpha_fleet_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    beta_fleet_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Fleet mode flag
    is_fleet_battle: bool = False


class LLMBattleRunner:
    """
    Orchestrates a battle between two LLM-controlled ships.

    Flow:
    1. Setup simulation with two ships
    2. Run simulation for decision_interval seconds
    3. Pause, deliver messages, query LLMs for decisions
    4. Apply commands, resume simulation
    5. Repeat until victory condition or checkpoint limit
    """

    def __init__(
        self,
        config: BattleConfig,
        alpha_config: LLMCaptainConfig,
        beta_config: LLMCaptainConfig,
        client: CaptainClient,
        fleet_config: Optional[BattleFleetConfig] = None,
    ):
        self.config = config
        self.alpha_config = alpha_config
        self.beta_config = beta_config
        self.client = client
        self.fleet_config = fleet_config

        self.simulation = None
        self.alpha_captain: Optional[LLMCaptain] = None
        self.beta_captain: Optional[LLMCaptain] = None
        self.communication: Optional[CommunicationChannel] = None
        self.fleet_communication: Optional[FleetCommunicationChannel] = None
        self.evaluator = VictoryEvaluator()
        self.recorder: Optional[BattleRecorder] = None

        self.checkpoint_count = 0
        self.decision_log: List[Dict[str, Any]] = []
        self.recording_file: Optional[str] = None

        # Track whether draw notifications have been sent (to avoid duplicates)
        self._alpha_draw_notified = False
        self._beta_draw_notified = False

        # Fleet mode: multiple ships and optional admirals
        self.is_fleet_mode = fleet_config is not None
        self.alpha_ships: Dict[str, Any] = {}  # ship_id -> ShipCombatState
        self.beta_ships: Dict[str, Any] = {}
        self.alpha_captains: Dict[str, LLMCaptain] = {}  # ship_id -> LLMCaptain
        self.beta_captains: Dict[str, LLMCaptain] = {}
        self.alpha_admiral: Optional[LLMAdmiral] = None
        self.beta_admiral: Optional[LLMAdmiral] = None

        # MCP controllers (replaces admirals when MCP-controlled)
        self.alpha_mcp: Optional[MCPController] = None
        self.beta_mcp: Optional[MCPController] = None

        # Shared chat for MCP inter-admiral communication
        self.mcp_chat: Optional[AdmiralChat] = None

        # Pre-checkpoint snapshot time offset for Admirals (seconds before checkpoint)
        self.admiral_pre_snapshot_offset = 15.0

    def setup_battle(self, fleet_data: Dict[str, Any]) -> None:
        """
        Initialize simulation and captains.

        Args:
            fleet_data: Ship specifications from fleet_ships.json
        """
        from ..simulation import CombatSimulation
        from ..physics import Vector3D

        # Create simulation
        self.simulation = CombatSimulation(
            time_step=1.0,
            decision_interval=self.config.decision_interval_s,
        )

        # Calculate positions
        half_dist = self.config.initial_distance_km * 1000 / 2  # meters
        offset = self.config.initial_offset_km * 1000  # meters

        # Create alpha ship (left side, facing right)
        alpha_ship = self._create_ship(
            ship_id="alpha",
            faction="alpha",
            ship_type=self.config.alpha_ship_type,
            fleet_data=fleet_data,
            position=Vector3D(-half_dist, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0),
        )
        self.simulation.add_ship(alpha_ship)

        # Create beta ship (right side, facing left)
        beta_ship = self._create_ship(
            ship_id="beta",
            faction="beta",
            ship_type=self.config.beta_ship_type,
            fleet_data=fleet_data,
            position=Vector3D(half_dist, offset, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0),
        )
        self.simulation.add_ship(beta_ship)

        # Store fleet_data on the runner
        self.fleet_data = fleet_data

        # Store ship types and fleet_data on captain configs before creating captains
        self.alpha_config.ship_type = self.config.alpha_ship_type
        self.beta_config.ship_type = self.config.beta_ship_type
        self.alpha_config.fleet_data = fleet_data
        self.beta_config.fleet_data = fleet_data

        # Create captains
        self.alpha_captain = LLMCaptain(self.alpha_config, self.client)
        self.beta_captain = LLMCaptain(self.beta_config, self.client)

        # Create communication channel
        self.communication = CommunicationChannel(
            alpha_name=self.alpha_config.name,
            alpha_ship=self.alpha_config.ship_name,
            beta_name=self.beta_config.name,
            beta_ship=self.beta_config.ship_name,
        )

        # Disable auto decision callback - we'll call manually
        self.simulation._decision_callback = None

        # Initialize battle recorder
        if self.config.record_battle:
            self.recorder = BattleRecorder()
            self.recorder.start_recording(
                battle_config=self.config,
                alpha_config=self.alpha_config,
                beta_config=self.beta_config,
                alpha_ship=alpha_ship,
                beta_ship=beta_ship,
            )
            # Register event callback to capture combat events
            self.simulation.add_event_callback(self._handle_simulation_event)

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"LLM BATTLE: {self.alpha_config.ship_name} vs {self.beta_config.ship_name}")
            print(f"{'='*60}")
            print(f"Distance: {self.config.initial_distance_km} km")
            print(f"Alpha: {self.alpha_config.name}")
            print(f"Beta: {self.beta_config.name}")
            print(f"{'='*60}\n")

    def _create_ship(
        self,
        ship_id: str,
        faction: str,
        ship_type: str,
        fleet_data: Dict[str, Any],
        position: Any,
        velocity: Any,
        forward: Any,
    ) -> Any:
        """Create a ship from fleet data."""
        from ..simulation import create_ship_from_fleet_data

        return create_ship_from_fleet_data(
            ship_id=ship_id,
            ship_type=ship_type,
            faction=faction,
            fleet_data=fleet_data,
            position=position,
            velocity=velocity,
            forward=forward,
        )

    def setup_fleet_battle(self, fleet_data: Dict[str, Any]) -> None:
        """
        Initialize simulation with fleet configuration (multi-ship, Admirals).

        Args:
            fleet_data: Ship specifications from fleet_ships.json
        """
        from ..simulation import CombatSimulation
        from ..physics import Vector3D

        if not self.fleet_config:
            raise ValueError("Fleet config required for fleet battles")

        # Create simulation
        self.simulation = CombatSimulation(
            time_step=1.0,
            decision_interval=self.fleet_config.decision_interval_s,
        )

        # Calculate base positions
        half_dist = self.fleet_config.initial_distance_km * 1000 / 2  # meters

        # Store fleet_data on the runner
        self.fleet_data = fleet_data

        # Create alpha fleet ships
        alpha_ship_ids = []
        for i, ship_config in enumerate(self.fleet_config.alpha_fleet.ships):
            # Use config position or calculate based on index
            if ship_config.position:
                pos = Vector3D(
                    ship_config.position.get('x', -half_dist) * 1000,
                    ship_config.position.get('y', i * 2000) * 1000,
                    ship_config.position.get('z', 0) * 1000,
                )
            else:
                # Spread ships vertically
                pos = Vector3D(-half_dist, i * 5000, 0)

            if ship_config.velocity:
                vel = Vector3D(
                    ship_config.velocity.get('x', 0) * 1000,
                    ship_config.velocity.get('y', 0) * 1000,
                    ship_config.velocity.get('z', 0) * 1000,
                )
            else:
                vel = Vector3D(0, 0, 0)

            ship = self._create_ship(
                ship_id=ship_config.ship_id,
                faction="alpha",
                ship_type=ship_config.ship_type,
                fleet_data=fleet_data,
                position=pos,
                velocity=vel,
                forward=Vector3D(1, 0, 0),
            )
            # Set ship name
            ship.name = ship_config.ship_name
            self.simulation.add_ship(ship)
            self.alpha_ships[ship_config.ship_id] = ship
            alpha_ship_ids.append(ship_config.ship_id)

            # Create captain for this ship
            captain_config = LLMCaptainConfig(
                name=ship_config.captain_name,
                model=ship_config.model,
                ship_name=ship_config.ship_name,
                ship_type=ship_config.ship_type,
                fleet_data=fleet_data,
                temperature=ship_config.temperature,
            )
            captain = LLMCaptain(captain_config, self.client)
            captain.ship_id = ship_config.ship_id
            captain.faction = "alpha"

            # Mark if captain has an Admiral
            if self.fleet_config.alpha_fleet.admiral:
                captain.has_admiral = True

            self.alpha_captains[ship_config.ship_id] = captain

        # Create beta fleet ships
        beta_ship_ids = []
        for i, ship_config in enumerate(self.fleet_config.beta_fleet.ships):
            # Use config position or calculate based on index
            if ship_config.position:
                pos = Vector3D(
                    ship_config.position.get('x', half_dist) * 1000,
                    ship_config.position.get('y', i * 2000) * 1000,
                    ship_config.position.get('z', 0) * 1000,
                )
            else:
                # Spread ships vertically
                pos = Vector3D(half_dist, i * 5000, 0)

            if ship_config.velocity:
                vel = Vector3D(
                    ship_config.velocity.get('x', 0) * 1000,
                    ship_config.velocity.get('y', 0) * 1000,
                    ship_config.velocity.get('z', 0) * 1000,
                )
            else:
                vel = Vector3D(0, 0, 0)

            ship = self._create_ship(
                ship_id=ship_config.ship_id,
                faction="beta",
                ship_type=ship_config.ship_type,
                fleet_data=fleet_data,
                position=pos,
                velocity=vel,
                forward=Vector3D(-1, 0, 0),
            )
            # Set ship name
            ship.name = ship_config.ship_name
            self.simulation.add_ship(ship)
            self.beta_ships[ship_config.ship_id] = ship
            beta_ship_ids.append(ship_config.ship_id)

            # Create captain for this ship
            captain_config = LLMCaptainConfig(
                name=ship_config.captain_name,
                model=ship_config.model,
                ship_name=ship_config.ship_name,
                ship_type=ship_config.ship_type,
                fleet_data=fleet_data,
                temperature=ship_config.temperature,
            )
            captain = LLMCaptain(captain_config, self.client)
            captain.ship_id = ship_config.ship_id
            captain.faction = "beta"

            # Mark if captain has an Admiral
            if self.fleet_config.beta_fleet.admiral:
                captain.has_admiral = True

            self.beta_captains[ship_config.ship_id] = captain

        # Create shared chat for MCP communication (if any MCP fleet)
        if self.fleet_config.has_any_mcp():
            self.mcp_chat = AdmiralChat()

        # Create MCP controllers or Admirals based on configuration
        # MCP takes precedence over Admiral if both are configured
        if self.fleet_config.alpha_fleet.mcp and self.fleet_config.alpha_fleet.mcp.enabled:
            mcp_config = MCPControllerConfig(
                faction="alpha",
                name=self.fleet_config.alpha_fleet.mcp.name,
                command_timeout=self.fleet_config.alpha_fleet.mcp.command_timeout,
            )
            self.alpha_mcp = MCPController(
                config=mcp_config,
                fleet_data=fleet_data,
                chat=self.mcp_chat,
            )
        elif self.fleet_config.alpha_fleet.admiral:
            from .admiral import LLMAdmiral
            self.alpha_admiral = LLMAdmiral(
                config=self.fleet_config.alpha_fleet.admiral,
                faction="alpha",
                client=self.client,
                fleet_data=fleet_data,
            )

        if self.fleet_config.beta_fleet.mcp and self.fleet_config.beta_fleet.mcp.enabled:
            mcp_config = MCPControllerConfig(
                faction="beta",
                name=self.fleet_config.beta_fleet.mcp.name,
                command_timeout=self.fleet_config.beta_fleet.mcp.command_timeout,
            )
            self.beta_mcp = MCPController(
                config=mcp_config,
                fleet_data=fleet_data,
                chat=self.mcp_chat,
            )
        elif self.fleet_config.beta_fleet.admiral:
            from .admiral import LLMAdmiral
            self.beta_admiral = LLMAdmiral(
                config=self.fleet_config.beta_fleet.admiral,
                faction="beta",
                client=self.client,
                fleet_data=fleet_data,
            )

        # Create fleet communication channel and register ships
        self.fleet_communication = FleetCommunicationChannel()

        # Register alpha ships
        for ship_id in alpha_ship_ids:
            ship = self.alpha_ships[ship_id]
            captain = self.alpha_captains[ship_id]
            self.fleet_communication.register_ship(
                ship_id=ship_id,
                captain_name=captain.config.name,
                ship_name=ship.name,
                faction="alpha",
            )

        # Register beta ships
        for ship_id in beta_ship_ids:
            ship = self.beta_ships[ship_id]
            captain = self.beta_captains[ship_id]
            self.fleet_communication.register_ship(
                ship_id=ship_id,
                captain_name=captain.config.name,
                ship_name=ship.name,
                faction="beta",
            )

        # Register admirals/MCP controllers and set ship mappings
        alpha_name_to_id = {ship.name: ship_id for ship_id, ship in self.alpha_ships.items()}
        beta_name_to_id = {ship.name: ship_id for ship_id, ship in self.beta_ships.items()}

        if self.alpha_mcp:
            self.fleet_communication.register_admiral("alpha", self.alpha_mcp.name)
            self.alpha_mcp.set_ship_mapping(alpha_name_to_id)
        elif self.alpha_admiral:
            self.fleet_communication.register_admiral("alpha", self.alpha_admiral.name)
            self.alpha_admiral.set_ship_mapping(alpha_name_to_id)

        if self.beta_mcp:
            self.fleet_communication.register_admiral("beta", self.beta_mcp.name)
            self.beta_mcp.set_ship_mapping(beta_name_to_id)
        elif self.beta_admiral:
            self.fleet_communication.register_admiral("beta", self.beta_admiral.name)
            self.beta_admiral.set_ship_mapping(beta_name_to_id)

        # Disable auto decision callback - we'll call manually
        self.simulation._decision_callback = None

        # Initialize fleet battle recorder
        if self.config.record_battle:
            self.recorder = BattleRecorder()
            self.recorder.start_fleet_recording(
                fleet_config=self.fleet_config,
                battle_config=self.config,
                alpha_ships=self.alpha_ships,
                beta_ships=self.beta_ships,
                alpha_admiral=self.alpha_admiral,
                beta_admiral=self.beta_admiral,
            )
            # Register event callback to capture combat events
            self.simulation.add_event_callback(self._handle_simulation_event)

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"FLEET BATTLE: {self.fleet_config.battle_name}")
            print(f"{'='*60}")
            print(f"Distance: {self.fleet_config.initial_distance_km} km")
            print(f"\nAlpha Fleet ({len(self.alpha_ships)} ships):")
            for ship_id, ship in self.alpha_ships.items():
                captain = self.alpha_captains[ship_id]
                print(f"  - {ship.name} ({ship_id}): {captain.name}")
            if self.alpha_mcp:
                print(f"  MCP Controller: {self.alpha_mcp.name}")
            elif self.alpha_admiral:
                print(f"  Admiral: {self.alpha_admiral.name}")

            print(f"\nBeta Fleet ({len(self.beta_ships)} ships):")
            for ship_id, ship in self.beta_ships.items():
                captain = self.beta_captains[ship_id]
                print(f"  - {ship.name} ({ship_id}): {captain.name}")
            if self.beta_mcp:
                print(f"  MCP Controller: {self.beta_mcp.name}")
            elif self.beta_admiral:
                print(f"  Admiral: {self.beta_admiral.name}")
            print(f"{'='*60}\n")

    def run_battle(self, fleet_data: Dict[str, Any]) -> BattleResult:
        """
        Run a complete battle with LLM-controlled captains.

        Args:
            fleet_data: Ship specifications

        Returns:
            BattleResult with outcome and statistics
        """
        # Dispatch to fleet battle if in fleet mode
        if self.is_fleet_mode:
            return self.run_fleet_battle(fleet_data)

        self.setup_battle(fleet_data)

        # === PERSONALITY SELECTION PHASE ===
        if self.config.personality_selection:
            if self.config.verbose:
                print("\n=== PERSONALITY SELECTION PHASE ===")

            # Let each captain choose their personality
            if self.config.verbose:
                print(f"\n[{self.alpha_config.name}] Defining combat personality...")
            try:
                alpha_personality = self.alpha_captain.select_personality(
                    distance_km=self.config.initial_distance_km,
                    verbose=False,
                )
                if self.config.verbose:
                    desc = alpha_personality.get("personality_description", "")
                    if desc:
                        print(f"  {desc}")
            except Exception as e:
                if self.config.verbose:
                    print(f"  [ERROR] Personality selection failed: {e}")

            if self.config.verbose:
                print(f"\n[{self.beta_config.name}] Defining combat personality...")
            try:
                beta_personality = self.beta_captain.select_personality(
                    distance_km=self.config.initial_distance_km,
                    verbose=False,
                )
                if self.config.verbose:
                    desc = beta_personality.get("personality_description", "")
                    if desc:
                        print(f"  {desc}")
            except Exception as e:
                if self.config.verbose:
                    print(f"  [ERROR] Personality selection failed: {e}")

        while not self._is_battle_over():
            # === SIMULATION PHASE ===
            # Run for decision_interval seconds
            steps = int(self.config.decision_interval_s)
            for _ in range(steps):
                self.simulation.step()

                # Record sim frame if enabled
                if self.recorder and self.config.record_sim_trace:
                    self._record_sim_frame()

                if self._is_battle_over():
                    break

            if self._is_battle_over():
                break

            # === CHECKPOINT ===
            self.checkpoint_count += 1

            if self.config.verbose:
                print(f"\n=== CHECKPOINT {self.checkpoint_count} at T+{self.simulation.current_time:.0f}s ===")
                self._print_status()

            # Record checkpoint state
            if self.recorder:
                alpha = self.simulation.get_ship("alpha")
                beta = self.simulation.get_ship("beta")
                if alpha and beta:
                    dist = (alpha.position - beta.position).magnitude / 1000
                    self.recorder.record_checkpoint(
                        timestamp=self.simulation.current_time,
                        checkpoint_num=self.checkpoint_count,
                        alpha_state=self._build_ship_state_dict(alpha),
                        beta_state=self._build_ship_state_dict(beta),
                        distance_km=dist,
                    )

            # Phase 1: Deliver pending messages
            alpha_msgs = self.communication.deliver_messages("alpha")
            beta_msgs = self.communication.deliver_messages("beta")
            self.alpha_captain.receive_messages(alpha_msgs)
            self.beta_captain.receive_messages(beta_msgs)

            # Print delivered messages
            if self.config.verbose:
                for msg in alpha_msgs + beta_msgs:
                    print(f"  {msg.format_for_display()}")

            # Phase 2: Get decisions from both captains
            if self.config.verbose:
                print(f"\n[Alpha] {self.alpha_config.name} deciding...")
            alpha_commands = self.alpha_captain.decide("alpha", self.simulation)
            if self.config.verbose:
                print(f"  -> {self.alpha_captain.get_last_decision_summary()}")

            if self.config.verbose:
                print(f"[Beta] {self.beta_config.name} deciding...")
            beta_commands = self.beta_captain.decide("beta", self.simulation)
            if self.config.verbose:
                print(f"  -> {self.beta_captain.get_last_decision_summary()}")

            # Phase 3: Queue outgoing messages
            alpha_msg_data = self.alpha_captain.get_pending_message()
            beta_msg_data = self.beta_captain.get_pending_message()

            if alpha_msg_data:
                # Extract content and recipient from dict
                msg_content = alpha_msg_data.get("content", "") if isinstance(alpha_msg_data, dict) else str(alpha_msg_data)
                msg_recipient = alpha_msg_data.get("recipient", "ALL_ENEMIES") if isinstance(alpha_msg_data, dict) else "ALL_ENEMIES"

                # Only queue if recipient includes enemies (in 1v1, this is the other captain)
                if msg_recipient in ("ALL", "ALL_ENEMIES", "SPECIFIC"):
                    self.communication.queue_message(
                        "alpha", msg_content, self.simulation.current_time
                    )
                if self.recorder:
                    self.recorder.record_message(
                        timestamp=self.simulation.current_time,
                        sender_id="alpha",
                        sender_name=self.alpha_config.name,
                        ship_name=self.alpha_config.ship_name,
                        message=msg_content,
                    )
                if self.config.verbose:
                    recipient_tag = f" [{msg_recipient}]" if msg_recipient != "ALL_ENEMIES" else ""
                    print(f"  [{self.alpha_config.ship_name}] {self.alpha_config.name}{recipient_tag}: \"{msg_content}\"")

            if beta_msg_data:
                # Extract content and recipient from dict
                msg_content = beta_msg_data.get("content", "") if isinstance(beta_msg_data, dict) else str(beta_msg_data)
                msg_recipient = beta_msg_data.get("recipient", "ALL_ENEMIES") if isinstance(beta_msg_data, dict) else "ALL_ENEMIES"

                # Only queue if recipient includes enemies (in 1v1, this is the other captain)
                if msg_recipient in ("ALL", "ALL_ENEMIES", "SPECIFIC"):
                    self.communication.queue_message(
                        "beta", msg_content, self.simulation.current_time
                    )
                if self.recorder:
                    self.recorder.record_message(
                        timestamp=self.simulation.current_time,
                        sender_id="beta",
                        sender_name=self.beta_config.name,
                        ship_name=self.beta_config.ship_name,
                        message=msg_content,
                    )
                if self.config.verbose:
                    recipient_tag = f" [{msg_recipient}]" if msg_recipient != "ALL_ENEMIES" else ""
                    print(f"  [{self.beta_config.ship_name}] {self.beta_config.name}{recipient_tag}: \"{msg_content}\"")

            # Phase 4: Check surrender/draw
            if self.alpha_captain.has_surrendered:
                self.communication.alpha_surrendered = True
                if self.recorder:
                    self.recorder.record_surrender(
                        timestamp=self.simulation.current_time,
                        ship_id="alpha",
                        captain_name=self.alpha_config.name,
                    )
            if self.beta_captain.has_surrendered:
                self.communication.beta_surrendered = True
                if self.recorder:
                    self.recorder.record_surrender(
                        timestamp=self.simulation.current_time,
                        ship_id="beta",
                        captain_name=self.beta_config.name,
                    )
            # Handle alpha draw proposals/retractions
            if self.alpha_captain.has_proposed_draw and not self._alpha_draw_notified:
                # Queue message to notify beta captain
                self.communication.queue_message(
                    "alpha", "", self.simulation.current_time, MessageType.PROPOSE_DRAW
                )
                self._alpha_draw_notified = True
                if self.recorder:
                    self.recorder.record_draw_proposal(
                        timestamp=self.simulation.current_time,
                        ship_id="alpha",
                        captain_name=self.alpha_config.name,
                    )
                if self.config.verbose:
                    print(f"  [DRAW PROPOSED] {self.alpha_config.name} proposes draw")
            elif self.alpha_captain.has_retracted_draw and self._alpha_draw_notified:
                # Queue message to notify beta captain of retraction
                self.communication.queue_message(
                    "alpha", "", self.simulation.current_time, MessageType.RETRACT_DRAW
                )
                self._alpha_draw_notified = False
                self.alpha_captain.has_retracted_draw = False  # Clear the flag
                if self.config.verbose:
                    print(f"  [DRAW RETRACTED] {self.alpha_config.name} retracts draw")

            # Handle beta draw proposals/retractions
            if self.beta_captain.has_proposed_draw and not self._beta_draw_notified:
                # Queue message to notify alpha captain
                self.communication.queue_message(
                    "beta", "", self.simulation.current_time, MessageType.PROPOSE_DRAW
                )
                self._beta_draw_notified = True
                if self.recorder:
                    self.recorder.record_draw_proposal(
                        timestamp=self.simulation.current_time,
                        ship_id="beta",
                        captain_name=self.beta_config.name,
                    )
                if self.config.verbose:
                    print(f"  [DRAW PROPOSED] {self.beta_config.name} proposes draw")
            elif self.beta_captain.has_retracted_draw and self._beta_draw_notified:
                # Queue message to notify alpha captain of retraction
                self.communication.queue_message(
                    "beta", "", self.simulation.current_time, MessageType.RETRACT_DRAW
                )
                self._beta_draw_notified = False
                self.beta_captain.has_retracted_draw = False  # Clear the flag
                if self.config.verbose:
                    print(f"  [DRAW RETRACTED] {self.beta_config.name} retracts draw")

            if self.communication.is_battle_ended():
                break

            # Phase 5: Apply commands
            for cmd in alpha_commands:
                success = self.simulation.inject_command("alpha", cmd)
                if self.config.verbose and isinstance(cmd, dict) and cmd.get('type') == 'fire_at':
                    print(f"  [FIRE] Alpha {cmd.get('weapon_slot')} -> {'HIT' if success else 'FAILED'}")
            for cmd in beta_commands:
                success = self.simulation.inject_command("beta", cmd)
                if self.config.verbose and isinstance(cmd, dict) and cmd.get('type') == 'fire_at':
                    print(f"  [FIRE] Beta {cmd.get('weapon_slot')} -> {'HIT' if success else 'FAILED'}")

            # Log decision
            self._log_decision(alpha_commands, beta_commands)

            # Phase 6: Check checkpoint limit (skip in unlimited mode)
            if not self.config.unlimited_mode:
                if self.checkpoint_count >= self.config.max_checkpoints:
                    if self.config.verbose:
                        print(f"\n=== CHECKPOINT LIMIT REACHED ===")
                    break

        return self._evaluate_result()

    def run_fleet_battle(self, fleet_data: Dict[str, Any]) -> BattleResult:
        """
        Run a complete fleet battle with Admirals and multiple ships per side.

        Args:
            fleet_data: Ship specifications

        Returns:
            BattleResult with outcome and statistics
        """
        self.setup_fleet_battle(fleet_data)

        # Get decision interval from fleet config
        decision_interval = self.fleet_config.decision_interval_s if self.fleet_config else 30.0

        # === PERSONALITY SELECTION PHASE ===
        if self.config.personality_selection:
            if self.config.verbose:
                print("\n=== PERSONALITY SELECTION PHASE ===")

            # Let Admirals choose their personality first
            if self.alpha_admiral:
                if self.config.verbose:
                    print(f"\n[Admiral {self.alpha_admiral.name}] Defining command personality...")
                try:
                    personality = self.alpha_admiral.select_personality(
                        num_ships=len(self.alpha_ships),
                        verbose=False,
                    )
                    if self.config.verbose:
                        desc = personality.get("personality_description", "")
                        if desc:
                            print(f"  {desc}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"  [ERROR] Admiral personality selection failed: {e}")

            if self.beta_admiral:
                if self.config.verbose:
                    print(f"\n[Admiral {self.beta_admiral.name}] Defining command personality...")
                try:
                    personality = self.beta_admiral.select_personality(
                        num_ships=len(self.beta_ships),
                        verbose=False,
                    )
                    if self.config.verbose:
                        desc = personality.get("personality_description", "")
                        if desc:
                            print(f"  {desc}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"  [ERROR] Admiral personality selection failed: {e}")

            # Let each captain choose their personality
            for ship_id, captain in {**self.alpha_captains, **self.beta_captains}.items():
                if self.config.verbose:
                    print(f"\n[{captain.name}] Defining combat personality...")
                try:
                    personality = captain.select_personality(
                        distance_km=self.fleet_config.initial_distance_km if self.fleet_config else 500.0,
                        verbose=False,
                    )
                    if self.config.verbose:
                        desc = personality.get("personality_description", "")
                        if desc:
                            print(f"  {desc}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"  [ERROR] Personality selection failed: {e}")

        # Track time for Admiral pre-snapshots
        next_checkpoint_time = decision_interval

        while not self._is_fleet_battle_over():
            # === SIMULATION PHASE ===
            steps = int(decision_interval)
            for step_i in range(steps):
                current_time = self.simulation.current_time

                # Capture Admiral pre-snapshots at T-15s before checkpoint
                if current_time == next_checkpoint_time - self.admiral_pre_snapshot_offset:
                    self._capture_admiral_pre_snapshots()

                self.simulation.step()

                # Record sim frame if enabled
                if self.recorder and self.config.record_sim_trace:
                    self._record_sim_frame()

                if self._is_fleet_battle_over():
                    break

            if self._is_fleet_battle_over():
                break

            # === CHECKPOINT ===
            self.checkpoint_count += 1
            next_checkpoint_time = self.simulation.current_time + decision_interval

            if self.config.verbose:
                print(f"\n=== CHECKPOINT {self.checkpoint_count} at T+{self.simulation.current_time:.0f}s ===")
                self._print_fleet_status()

            # Phase 1: Admiral decisions (both sides)
            admiral_orders = {}  # ship_id -> list of AdmiralOrder
            if self.config.verbose and (self.alpha_admiral or self.beta_admiral):
                print("\n--- ADMIRAL DECISIONS ---")

            if self.alpha_admiral:
                # Filter out surrendered ships - Admiral shouldn't command them
                active_alpha_captains = [
                    c for c in self.alpha_captains.values()
                    if not getattr(self.alpha_ships.get(c.ship_id), 'is_surrendered', False)
                ]
                alpha_decision = self._get_admiral_decision(
                    self.alpha_admiral,
                    active_alpha_captains,
                    self.beta_admiral,
                )
                # Distribute orders to captains
                for order in alpha_decision.fleet_orders:
                    ship_id = self._find_ship_id_by_name(order.target_ship_id, "alpha")
                    if ship_id and ship_id in self.alpha_captains:
                        if ship_id not in admiral_orders:
                            admiral_orders[ship_id] = []
                        admiral_orders[ship_id].append(order)

                if self.config.verbose:
                    print(f"  [Alpha Admiral] {self.alpha_admiral.name}:")
                    if alpha_decision.fleet_directive:
                        print(f"    Directive: {alpha_decision.fleet_directive}")
                    for order in alpha_decision.fleet_orders:
                        # Show full order text with proper indentation
                        order_lines = order.order_text.strip().split('\n')
                        print(f"    -> {order.target_ship_id}:")
                        for line in order_lines:
                            print(f"         {line}")

                # Record admiral directive and orders
                if self.recorder:
                    if alpha_decision.fleet_directive:
                        self.recorder.record_admiral_directive(
                            timestamp=self.simulation.current_time,
                            admiral_name=self.alpha_admiral.name,
                            faction="alpha",
                            directive=alpha_decision.fleet_directive,
                        )
                    for order in alpha_decision.fleet_orders:
                        ship_id = self._find_ship_id_by_name(order.target_ship_id, "alpha")
                        ship_name = order.target_ship_id
                        self.recorder.record_admiral_order(
                            timestamp=self.simulation.current_time,
                            admiral_name=self.alpha_admiral.name,
                            ship_id=ship_id or order.target_ship_id,
                            ship_name=ship_name,
                            order_text=order.order_text,
                            priority=order.priority,
                            suggested_target=order.suggested_target,
                        )

            if self.beta_admiral:
                # Filter out surrendered ships - Admiral shouldn't command them
                active_beta_captains = [
                    c for c in self.beta_captains.values()
                    if not getattr(self.beta_ships.get(c.ship_id), 'is_surrendered', False)
                ]
                beta_decision = self._get_admiral_decision(
                    self.beta_admiral,
                    active_beta_captains,
                    self.alpha_admiral,
                )
                # Distribute orders to captains
                for order in beta_decision.fleet_orders:
                    ship_id = self._find_ship_id_by_name(order.target_ship_id, "beta")
                    if ship_id and ship_id in self.beta_captains:
                        if ship_id not in admiral_orders:
                            admiral_orders[ship_id] = []
                        admiral_orders[ship_id].append(order)

                if self.config.verbose:
                    print(f"  [Beta Admiral] {self.beta_admiral.name}:")
                    if beta_decision.fleet_directive:
                        print(f"    Directive: {beta_decision.fleet_directive}")
                    for order in beta_decision.fleet_orders:
                        # Show full order text with proper indentation
                        order_lines = order.order_text.strip().split('\n')
                        print(f"    -> {order.target_ship_id}:")
                        for line in order_lines:
                            print(f"         {line}")

                # Record admiral directive and orders
                if self.recorder:
                    if beta_decision.fleet_directive:
                        self.recorder.record_admiral_directive(
                            timestamp=self.simulation.current_time,
                            admiral_name=self.beta_admiral.name,
                            faction="beta",
                            directive=beta_decision.fleet_directive,
                        )
                    for order in beta_decision.fleet_orders:
                        ship_id = self._find_ship_id_by_name(order.target_ship_id, "beta")
                        ship_name = order.target_ship_id
                        self.recorder.record_admiral_order(
                            timestamp=self.simulation.current_time,
                            admiral_name=self.beta_admiral.name,
                            ship_id=ship_id or order.target_ship_id,
                            ship_name=ship_name,
                            order_text=order.order_text,
                            priority=order.priority,
                            suggested_target=order.suggested_target,
                        )

            # Phase 2: Deliver pending messages to captains
            if self.fleet_communication:
                for ship_id in list(self.alpha_captains.keys()) + list(self.beta_captains.keys()):
                    msgs = self.fleet_communication.deliver_messages(ship_id)
                    captain = self.alpha_captains.get(ship_id) or self.beta_captains.get(ship_id)
                    if captain and msgs:
                        captain.receive_messages(msgs)
                        if self.config.verbose:
                            for msg in msgs:
                                print(f"  {msg.format_for_display()}")

            # Phase 3: Captain decisions (with Admiral orders and discussion)
            all_commands = {}  # ship_id -> list of commands
            if self.config.verbose:
                print("\n--- CAPTAIN DECISIONS ---")

            # Process all captains
            all_captains = [
                (ship_id, captain, "alpha")
                for ship_id, captain in self.alpha_captains.items()
            ] + [
                (ship_id, captain, "beta")
                for ship_id, captain in self.beta_captains.items()
            ]

            for ship_id, captain, faction in all_captains:
                # Skip destroyed or surrendered ships
                ship = self.simulation.get_ship(ship_id)
                if not ship or ship.is_destroyed or getattr(ship, 'is_surrendered', False):
                    continue

                # Clear previous Admiral context and deliver new orders
                captain.clear_admiral_context()
                if ship_id in admiral_orders:
                    orders = admiral_orders[ship_id]
                    # Also include fleet directive if Admiral exists
                    admiral = self.alpha_admiral if faction == "alpha" else self.beta_admiral
                    if admiral and hasattr(admiral, 'last_directive'):
                        directive = admiral.last_directive
                    else:
                        directive = None
                    captain.receive_admiral_orders(orders, directive)

                if self.config.verbose:
                    print(f"  [{captain.ship_name}] {captain.name} deciding...")

                # Get decision (may include discuss_with_admiral request)
                commands = self._get_captain_decision_with_discussion(
                    ship_id, captain, faction
                )
                all_commands[ship_id] = commands

                if self.config.verbose:
                    print(f"    -> {self._get_ship_status_line(ship_id, commands)}")

                # Record captain decision
                if self.recorder:
                    self._record_captain_decision(ship_id, captain, commands)

            # Phase 4: Handle immediate messaging (captain to captain)
            # Process any pending broadcast or enemy messages
            self._handle_immediate_messaging()

            # Phase 5: Check surrender/draw
            self._check_fleet_surrender_draw()

            if self._is_fleet_battle_over():
                break

            # Phase 6: Apply commands
            for ship_id, commands in all_commands.items():
                for cmd in commands:
                    # Filter out discussion markers (handled above)
                    if isinstance(cmd, dict) and cmd.get('type') == 'discuss_with_admiral':
                        continue
                    success = self.simulation.inject_command(ship_id, cmd)
                    if self.config.verbose and isinstance(cmd, dict) and cmd.get('type') == 'fire_at':
                        print(f"    [FIRE] {ship_id} {cmd.get('weapon_slot')} -> {'HIT' if success else 'FAILED'}")

            # Log decision
            self._log_fleet_decision(all_commands)

            # Phase 7: Check limits
            if not self.config.unlimited_mode:
                max_checkpoints = self.fleet_config.max_checkpoints if self.fleet_config and hasattr(self.fleet_config, 'max_checkpoints') else self.config.max_checkpoints
                if self.checkpoint_count >= max_checkpoints:
                    if self.config.verbose:
                        print(f"\n=== CHECKPOINT LIMIT REACHED ===")
                    break

        return self._evaluate_fleet_result()

    async def run_fleet_battle_async(self, fleet_data: Dict[str, Any]) -> BattleResult:
        """
        Run a fleet battle with async support for MCP controllers.

        This method should be used when any fleet is MCP-controlled.
        Non-MCP fleets still use synchronous LLM calls.

        Args:
            fleet_data: Ship specifications

        Returns:
            BattleResult with outcome and statistics
        """
        self.setup_fleet_battle(fleet_data)

        # Get decision interval from fleet config
        decision_interval = self.fleet_config.decision_interval_s if self.fleet_config else 30.0

        # Skip personality selection for MCP-controlled fleets
        if self.config.personality_selection:
            if self.config.verbose:
                print("\n=== PERSONALITY SELECTION PHASE ===")

            # Let non-MCP Admirals choose personality
            if self.alpha_admiral and not self.alpha_mcp:
                if self.config.verbose:
                    print(f"\n[Admiral {self.alpha_admiral.name}] Defining command personality...")
                try:
                    personality = self.alpha_admiral.select_personality(
                        num_ships=len(self.alpha_ships),
                        verbose=False,
                    )
                    if self.config.verbose:
                        desc = personality.get("personality_description", "")
                        if desc:
                            print(f"  {desc}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"  [ERROR] Admiral personality selection failed: {e}")

            if self.beta_admiral and not self.beta_mcp:
                if self.config.verbose:
                    print(f"\n[Admiral {self.beta_admiral.name}] Defining command personality...")
                try:
                    personality = self.beta_admiral.select_personality(
                        num_ships=len(self.beta_ships),
                        verbose=False,
                    )
                    if self.config.verbose:
                        desc = personality.get("personality_description", "")
                        if desc:
                            print(f"  {desc}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"  [ERROR] Admiral personality selection failed: {e}")

            # Skip captain personality selection for MCP-controlled fleets
            # (MCP controls ships directly, bypassing captains)
            for ship_id, captain in {**self.alpha_captains, **self.beta_captains}.items():
                # Skip if this fleet is MCP-controlled
                faction = captain.faction if hasattr(captain, 'faction') else None
                if (faction == "alpha" and self.alpha_mcp) or (faction == "beta" and self.beta_mcp):
                    continue

                if self.config.verbose:
                    print(f"\n[{captain.name}] Defining combat personality...")
                try:
                    personality = captain.select_personality(
                        distance_km=self.fleet_config.initial_distance_km if self.fleet_config else 500.0,
                        verbose=False,
                    )
                    if self.config.verbose:
                        desc = personality.get("personality_description", "")
                        if desc:
                            print(f"  {desc}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"  [ERROR] Personality selection failed: {e}")

        # Track time for Admiral pre-snapshots
        next_checkpoint_time = decision_interval

        # Advance chat turn at start
        if self.mcp_chat:
            self.mcp_chat.new_turn()

        while not self._is_fleet_battle_over():
            # === SIMULATION PHASE ===
            steps = int(decision_interval)
            for step_i in range(steps):
                current_time = self.simulation.current_time

                # Capture Admiral pre-snapshots at T-15s before checkpoint
                if current_time == next_checkpoint_time - self.admiral_pre_snapshot_offset:
                    self._capture_admiral_pre_snapshots()

                self.simulation.step()

                # Record sim frame if enabled
                if self.recorder and self.config.record_sim_trace:
                    self._record_sim_frame()

                if self._is_fleet_battle_over():
                    break

            if self._is_fleet_battle_over():
                break

            # === CHECKPOINT ===
            self.checkpoint_count += 1
            next_checkpoint_time = self.simulation.current_time + decision_interval

            if self.config.verbose:
                print(f"\n=== CHECKPOINT {self.checkpoint_count} at T+{self.simulation.current_time:.0f}s ===")
                self._print_fleet_status()

            # Advance chat turn
            if self.mcp_chat:
                self.mcp_chat.new_turn()

            # === MCP/ADMIRAL DECISION PHASE ===
            admiral_orders = {}  # ship_id -> list of AdmiralOrder
            all_commands = {}    # ship_id -> list of commands

            # Handle alpha fleet
            if self.alpha_mcp:
                # MCP-controlled: get commands from MCP client
                if self.config.verbose:
                    print(f"\n--- MCP COMMAND PHASE (Alpha: {self.alpha_mcp.name}) ---")
                    print(f"  Waiting for MCP client commands...")

                mcp_commands = await self.alpha_mcp.get_commands(
                    self.simulation,
                    list(self.alpha_captains.values()),
                )

                if self.config.verbose:
                    print(f"  Received {len(mcp_commands)} commands")

                # Apply MCP commands directly to simulation
                results = apply_mcp_commands_to_simulation(
                    mcp_commands, self.simulation, "alpha"
                )

                if self.config.verbose and results.get("applied"):
                    for cmd_result in results["applied"]:
                        print(f"    Applied: {cmd_result}")

            elif self.alpha_admiral:
                # LLM Admiral: use existing logic
                if self.config.verbose:
                    print("\n--- ADMIRAL DECISIONS ---")

                active_alpha_captains = [
                    c for c in self.alpha_captains.values()
                    if not getattr(self.alpha_ships.get(c.ship_id), 'is_surrendered', False)
                ]
                alpha_decision = self._get_admiral_decision(
                    self.alpha_admiral,
                    active_alpha_captains,
                    self.beta_admiral,
                )
                for order in alpha_decision.fleet_orders:
                    ship_id = self._find_ship_id_by_name(order.target_ship_id, "alpha")
                    if ship_id and ship_id in self.alpha_captains:
                        if ship_id not in admiral_orders:
                            admiral_orders[ship_id] = []
                        admiral_orders[ship_id].append(order)

                if self.config.verbose:
                    print(f"  [Alpha Admiral] {self.alpha_admiral.name}:")
                    if alpha_decision.fleet_directive:
                        print(f"    Directive: {alpha_decision.fleet_directive}")
                    for order in alpha_decision.fleet_orders:
                        order_lines = order.order_text.strip().split('\n')
                        print(f"    -> {order.target_ship_id}:")
                        for line in order_lines:
                            print(f"         {line}")

            # Handle beta fleet
            if self.beta_mcp:
                # MCP-controlled: get commands from MCP client
                if self.config.verbose:
                    print(f"\n--- MCP COMMAND PHASE (Beta: {self.beta_mcp.name}) ---")
                    print(f"  Waiting for MCP client commands...")

                mcp_commands = await self.beta_mcp.get_commands(
                    self.simulation,
                    list(self.beta_captains.values()),
                )

                if self.config.verbose:
                    print(f"  Received {len(mcp_commands)} commands")

                # Apply MCP commands directly to simulation
                results = apply_mcp_commands_to_simulation(
                    mcp_commands, self.simulation, "beta"
                )

                if self.config.verbose and results.get("applied"):
                    for cmd_result in results["applied"]:
                        print(f"    Applied: {cmd_result}")

            elif self.beta_admiral:
                # LLM Admiral: use existing logic
                active_beta_captains = [
                    c for c in self.beta_captains.values()
                    if not getattr(self.beta_ships.get(c.ship_id), 'is_surrendered', False)
                ]
                beta_decision = self._get_admiral_decision(
                    self.beta_admiral,
                    active_beta_captains,
                    self.alpha_admiral,
                )
                for order in beta_decision.fleet_orders:
                    ship_id = self._find_ship_id_by_name(order.target_ship_id, "beta")
                    if ship_id and ship_id in self.beta_captains:
                        if ship_id not in admiral_orders:
                            admiral_orders[ship_id] = []
                        admiral_orders[ship_id].append(order)

                if self.config.verbose:
                    print(f"  [Beta Admiral] {self.beta_admiral.name}:")
                    if beta_decision.fleet_directive:
                        print(f"    Directive: {beta_decision.fleet_directive}")
                    for order in beta_decision.fleet_orders:
                        order_lines = order.order_text.strip().split('\n')
                        print(f"    -> {order.target_ship_id}:")
                        for line in order_lines:
                            print(f"         {line}")

            # === MESSAGE BRIDGE: MCP <-> LLM Admiral ===
            # Deliver messages between MCP chat system and LLM Admiral messaging
            if self.mcp_chat:
                # MCP -> LLM Admiral: Deliver pending messages from MCP to LLM admirals
                if self.alpha_mcp and self.beta_admiral:
                    pending_for_beta = self.mcp_chat.get_pending_messages("beta")
                    for msg in pending_for_beta:
                        self.beta_admiral.receive_enemy_admiral_message(msg.content)
                        if self.config.verbose:
                            print(f"  [MSG] Alpha MCP -> Beta Admiral: \"{msg.content}\"")

                if self.beta_mcp and self.alpha_admiral:
                    pending_for_alpha = self.mcp_chat.get_pending_messages("alpha")
                    for msg in pending_for_alpha:
                        self.alpha_admiral.receive_enemy_admiral_message(msg.content)
                        if self.config.verbose:
                            print(f"  [MSG] Beta MCP -> Alpha Admiral: \"{msg.content}\"")

                # LLM Admiral -> MCP: Add LLM admiral messages to MCP chat
                if self.alpha_admiral and self.beta_mcp:
                    alpha_msg = self.alpha_admiral.get_pending_enemy_message()
                    if alpha_msg:
                        self.mcp_chat.send_message("alpha", alpha_msg, self.simulation.current_time)
                        if self.config.verbose:
                            print(f"  [MSG] Alpha Admiral -> Beta MCP: \"{alpha_msg}\"")

                if self.beta_admiral and self.alpha_mcp:
                    beta_msg = self.beta_admiral.get_pending_enemy_message()
                    if beta_msg:
                        self.mcp_chat.send_message("beta", beta_msg, self.simulation.current_time)
                        if self.config.verbose:
                            print(f"  [MSG] Beta Admiral -> Alpha MCP: \"{beta_msg}\"")

            # === CAPTAIN DECISIONS (only for non-MCP fleets) ===
            if self.config.verbose and (not self.alpha_mcp or not self.beta_mcp):
                print("\n--- CAPTAIN DECISIONS ---")

            all_captains = [
                (ship_id, captain, "alpha")
                for ship_id, captain in self.alpha_captains.items()
            ] + [
                (ship_id, captain, "beta")
                for ship_id, captain in self.beta_captains.items()
            ]

            for ship_id, captain, faction in all_captains:
                # Skip if this fleet is MCP-controlled
                if (faction == "alpha" and self.alpha_mcp) or (faction == "beta" and self.beta_mcp):
                    continue

                # Skip destroyed or surrendered ships
                ship = self.simulation.get_ship(ship_id)
                if not ship or ship.is_destroyed or getattr(ship, 'is_surrendered', False):
                    continue

                # Clear previous Admiral context and deliver new orders
                captain.clear_admiral_context()
                if ship_id in admiral_orders:
                    orders = admiral_orders[ship_id]
                    admiral = self.alpha_admiral if faction == "alpha" else self.beta_admiral
                    if admiral and hasattr(admiral, 'last_directive'):
                        directive = admiral.last_directive
                    else:
                        directive = None
                    captain.receive_admiral_orders(orders, directive)

                if self.config.verbose:
                    print(f"  [{captain.ship_name}] {captain.name} deciding...")

                commands = self._get_captain_decision_with_discussion(
                    ship_id, captain, faction
                )
                all_commands[ship_id] = commands

                if self.config.verbose:
                    print(f"    -> {self._get_ship_status_line(ship_id, commands)}")

            # Handle immediate messaging
            self._handle_immediate_messaging()

            # Check surrender/draw (including MCP)
            if self.alpha_mcp:
                if self.alpha_mcp.has_surrendered:
                    # Mark all alpha ships as surrendered
                    for ship_id in self.alpha_ships:
                        ship = self.simulation.get_ship(ship_id)
                        if ship:
                            ship.is_surrendered = True
                    if self.config.verbose:
                        print(f"  [SURRENDER] {self.alpha_mcp.name} surrenders")

            if self.beta_mcp:
                if self.beta_mcp.has_surrendered:
                    # Mark all beta ships as surrendered
                    for ship_id in self.beta_ships:
                        ship = self.simulation.get_ship(ship_id)
                        if ship:
                            ship.is_surrendered = True
                    if self.config.verbose:
                        print(f"  [SURRENDER] {self.beta_mcp.name} surrenders")

            # Check for mutual draw (MCP)
            if self.alpha_mcp and self.beta_mcp:
                if self.alpha_mcp.has_proposed_draw and self.beta_mcp.has_accepted_draw:
                    if self.config.verbose:
                        print("  [DRAW ACCEPTED] Mutual draw agreed")
                    break
                if self.beta_mcp.has_proposed_draw and self.alpha_mcp.has_accepted_draw:
                    if self.config.verbose:
                        print("  [DRAW ACCEPTED] Mutual draw agreed")
                    break

            self._check_fleet_surrender_draw()

            if self._is_fleet_battle_over():
                break

            # Apply captain commands (non-MCP only)
            for ship_id, commands in all_commands.items():
                for cmd in commands:
                    if isinstance(cmd, dict) and cmd.get('type') == 'discuss_with_admiral':
                        continue
                    success = self.simulation.inject_command(ship_id, cmd)
                    if self.config.verbose and isinstance(cmd, dict) and cmd.get('type') == 'fire_at':
                        print(f"    [FIRE] {ship_id} {cmd.get('weapon_slot')} -> {'HIT' if success else 'FAILED'}")

            # Log decision
            self._log_fleet_decision(all_commands)

            # Check limits
            if not self.config.unlimited_mode:
                max_checkpoints = self.fleet_config.max_checkpoints if self.fleet_config and hasattr(self.fleet_config, 'max_checkpoints') else self.config.max_checkpoints
                if self.checkpoint_count >= max_checkpoints:
                    if self.config.verbose:
                        print(f"\n=== CHECKPOINT LIMIT REACHED ===")
                    break

        return self._evaluate_fleet_result()

    def _capture_admiral_pre_snapshots(self) -> None:
        """Capture pre-checkpoint snapshots for Admirals at T-15s."""
        if self.alpha_admiral:
            self.alpha_admiral.capture_pre_snapshot(
                self.simulation,
                list(self.alpha_captains.values())
            )
        if self.beta_admiral:
            self.beta_admiral.capture_pre_snapshot(
                self.simulation,
                list(self.beta_captains.values())
            )

    def _get_admiral_decision(
        self,
        admiral: LLMAdmiral,
        friendly_captains: List[LLMCaptain],
        enemy_admiral: Optional[LLMAdmiral],
    ) -> Any:
        """Get Admiral's decision for the checkpoint."""
        return admiral.decide(
            simulation=self.simulation,
            captains=friendly_captains,
            enemy_admiral=enemy_admiral,
        )

    def _find_ship_id_by_name(self, name: str, faction: str) -> Optional[str]:
        """Find ship_id by ship name within a faction."""
        ships = self.alpha_ships if faction == "alpha" else self.beta_ships
        for ship_id, ship in ships.items():
            if getattr(ship, 'name', ship_id) == name or ship_id == name:
                return ship_id
        return None

    def _get_captain_decision_with_discussion(
        self,
        ship_id: str,
        captain: LLMCaptain,
        faction: str,
    ) -> List[Any]:
        """
        Get captain's decision, handling Admiral discussions.

        If captain requests to discuss with Admiral, this handles
        the synchronous discussion loop.
        """
        admiral = self.alpha_admiral if faction == "alpha" else self.beta_admiral

        # Get initial decision
        commands = captain.decide(ship_id, self.simulation)

        # Separate tactical commands from discussion requests
        # Tactical commands are things like maneuvers, fire commands, etc.
        from ..simulation import Maneuver

        def is_tactical_command(cmd):
            # Maneuver objects are tactical commands
            if isinstance(cmd, Maneuver):
                return True
            # Dict commands - check if not a discussion/response type
            if isinstance(cmd, dict):
                cmd_type = cmd.get('type', '')
                return cmd_type not in ('discuss_with_admiral', 'discussion_limit_reached', 'respond_to_orders')
            # Other object types (e.g., dataclasses) are tactical commands
            return True

        initial_tactical_commands = [c for c in commands if is_tactical_command(c)]

        # Check if captain requested Admiral discussion
        for cmd in commands:
            if isinstance(cmd, dict) and cmd.get('type') == 'discuss_with_admiral':
                if not admiral:
                    continue

                question = cmd.get('question', '')
                exchange = cmd.get('exchange_number', 1)

                if self.config.verbose:
                    print(f"    [DISCUSS] {captain.name} asks Admiral:")
                    # Print full question with indentation
                    for line in question.split('\n'):
                        print(f"      {line}")

                # Get Admiral's response
                response = admiral.respond_to_captain(
                    captain_ship_name=captain.ship_name,
                    question=question,
                    simulation=self.simulation,
                )

                if self.config.verbose:
                    print(f"    [ADMIRAL] {admiral.name} responds:")
                    # Print full response with indentation
                    for line in response.split('\n'):
                        print(f"      {line}")

                # Record the discussion
                if self.recorder:
                    self.recorder.record_captain_admiral_discussion(
                        timestamp=self.simulation.current_time,
                        ship_id=ship_id,
                        captain_name=captain.name,
                        admiral_name=admiral.name,
                        captain_question=question,
                        admiral_response=response,
                        exchange_number=exchange,
                    )

                # After discussion, captain should make a new decision with the Admiral's response
                # Inject the response into captain's context
                from .admiral import AdmiralOrder
                clarification_order = AdmiralOrder(
                    target_ship_id=ship_id,
                    target_ship_name=captain.ship_name,
                    order_text=f"[CLARIFICATION] {response}",
                    priority="NORMAL",
                    suggested_target=None,
                )
                captain.admiral_orders.append(clarification_order)

                # Get a new decision from the captain with the clarification
                new_commands = captain.decide(ship_id, self.simulation)

                # Filter out discussion requests from new commands
                new_tactical_commands = [c for c in new_commands if is_tactical_command(c)]

                # Use new tactical commands if any, otherwise fall back to initial tactical commands
                if new_tactical_commands:
                    commands = new_tactical_commands
                elif initial_tactical_commands:
                    # Captain only acknowledged but didn't issue new commands
                    # Use their initial tactical commands
                    commands = initial_tactical_commands
                else:
                    # No commands at all - force captain to decide again with explicit reminder
                    if self.config.verbose:
                        print(f"    [RETRY] {captain.name} must issue tactical commands...")

                    # Add a forceful reminder to the captain's orders
                    from .admiral import AdmiralOrder
                    force_order = AdmiralOrder(
                        target_ship_id=ship_id,
                        target_ship_name=captain.ship_name,
                        order_text="[CRITICAL] You MUST call set_maneuver, set_primary_target, and set_weapons_order NOW. Responding with words only will cause your ship to DRIFT and DIE. CALL THE TOOLS!",
                        priority="CRITICAL",
                        suggested_target=None,
                    )
                    captain.admiral_orders.append(force_order)

                    # Try one more time
                    retry_commands = captain.decide(ship_id, self.simulation)
                    retry_tactical = [c for c in retry_commands if is_tactical_command(c)]

                    if retry_tactical:
                        commands = retry_tactical
                    else:
                        # Still nothing - warn and continue with empty commands (will drift)
                        if self.config.verbose:
                            print(f"    [WARNING] {captain.name} still issued no tactical commands after retry")
                        commands = []

                break  # Only process one discussion per checkpoint

        # Filter out any remaining discussion dicts
        commands = [c for c in commands if not (isinstance(c, dict) and c.get('type') in ('discuss_with_admiral', 'discussion_limit_reached'))]

        return commands

    def _handle_immediate_messaging(self) -> None:
        """Handle immediate captain-to-captain messaging within checkpoint."""
        if not self.fleet_communication:
            return

        # Process any immediate messages (broadcasts, captain-to-captain)
        # Deliver to all relevant recipients
        for ship_id in list(self.alpha_captains.keys()) + list(self.beta_captains.keys()):
            immediate_msgs = self.fleet_communication.deliver_immediate_messages(ship_id)
            captain = self.alpha_captains.get(ship_id) or self.beta_captains.get(ship_id)
            if captain and immediate_msgs:
                captain.receive_messages(immediate_msgs)
                if self.config.verbose:
                    for msg in immediate_msgs:
                        print(f"  [IMMEDIATE] {msg.format_for_display()}")

    def _check_fleet_surrender_draw(self) -> None:
        """Check and handle surrenders and draws in fleet mode."""
        # Check each captain for surrender
        for ship_id, captain in {**self.alpha_captains, **self.beta_captains}.items():
            if captain.has_surrendered:
                ship = self.simulation.get_ship(ship_id)
                if ship and not getattr(ship, 'is_surrendered', False):
                    ship.is_surrendered = True
                    ship.current_maneuver = None  # Stop maneuvering
                    if self.config.verbose:
                        print(f"  [SURRENDER] {captain.ship_name} ({captain.name}) surrenders")

        # Handle Admiral-level draw proposals
        if self.alpha_admiral and hasattr(self.alpha_admiral, 'proposed_draw') and self.alpha_admiral.proposed_draw:
            if not self._alpha_draw_notified:
                self._alpha_draw_notified = True
                if self.config.verbose:
                    print(f"  [DRAW PROPOSED] Admiral {self.alpha_admiral.name} proposes fleet draw")

        if self.beta_admiral and hasattr(self.beta_admiral, 'proposed_draw') and self.beta_admiral.proposed_draw:
            if not self._beta_draw_notified:
                self._beta_draw_notified = True
                if self.config.verbose:
                    print(f"  [DRAW PROPOSED] Admiral {self.beta_admiral.name} proposes fleet draw")

    def _print_fleet_status(self) -> None:
        """Print current fleet battle status."""
        from ..combat import HitLocation

        def get_armor(ship, loc):
            sec = ship.armor.get_section(loc) if ship.armor else None
            return sec.thickness_cm if sec else 0

        def get_closest_enemy_dist(ship, enemy_ships):
            """Get distance to closest enemy in km."""
            if not enemy_ships:
                return None
            min_dist = float('inf')
            for enemy in enemy_ships.values():
                if enemy.is_destroyed:
                    continue
                dist = (ship.position - enemy.position).magnitude / 1000  # km
                if dist < min_dist:
                    min_dist = dist
            return min_dist if min_dist != float('inf') else None

        def format_ship_status(ship, enemy_ships):
            status = "DESTROYED" if ship.is_destroyed else (
                "SURRENDERED" if getattr(ship, 'is_surrendered', False) else "ACTIVE"
            )
            if ship.is_destroyed:
                return f"{ship.name}: DESTROYED"

            # Get armor values
            nose = get_armor(ship, HitLocation.NOSE)
            lat = get_armor(ship, HitLocation.LATERAL)
            tail = get_armor(ship, HitLocation.TAIL)

            # Get heat
            heat = ship.thermal_system.heat_percent if ship.thermal_system else 0

            # Get position in km (rounded)
            pos_km_x = ship.position.x / 1000
            pos_km_y = ship.position.y / 1000
            pos_km_z = ship.position.z / 1000
            pos_str = f"({pos_km_x:.0f},{pos_km_y:.0f},{pos_km_z:.0f})"

            # Get velocity in km/s (vector and magnitude)
            vel_kps_x = ship.velocity.x / 1000
            vel_kps_y = ship.velocity.y / 1000
            vel_kps_z = ship.velocity.z / 1000
            speed_kps = ship.velocity.magnitude / 1000
            vel_str = f"({vel_kps_x:.1f},{vel_kps_y:.1f},{vel_kps_z:.1f}) {speed_kps:.1f}km/s"

            return (
                f"{ship.name}: {ship.hull_integrity:.0f}% hull, {heat:.0f}% heat, "
                f"armor: N{nose:.0f}/L{lat:.0f}/T{tail:.0f}cm, "
                f"pos: {pos_str}km, vel: {vel_str}, shots: {ship.shots_fired}, hits: {ship.hits_scored}, {status}"
            )

        print("  --- Fleet Status ---")

        # Alpha fleet
        print("  Alpha Fleet:")
        for ship_id, ship in self.alpha_ships.items():
            print(f"    {format_ship_status(ship, self.beta_ships)}")

        # Beta fleet
        print("  Beta Fleet:")
        for ship_id, ship in self.beta_ships.items():
            print(f"    {format_ship_status(ship, self.alpha_ships)}")

    def _get_ship_status_line(self, ship_id: str, commands: Optional[List[Any]] = None) -> str:
        """Get a one-line status showing ship's decided maneuver, target, and weapons.

        Args:
            ship_id: Ship to get status for
            commands: Optional list of commands just issued - shows these instead of current state
        """
        from ..simulation import Maneuver

        ship = self.simulation.get_ship(ship_id)
        if not ship:
            return "Ship not found"

        parts = []
        commands = commands or []

        # Find maneuver from commands, fall back to current
        new_maneuver = None
        for cmd in commands:
            if isinstance(cmd, Maneuver):
                new_maneuver = cmd
                break

        if new_maneuver:
            maneuver_type = new_maneuver.maneuver_type.name
            throttle = new_maneuver.throttle * 100
            parts.append(f"{maneuver_type} @ {throttle:.0f}%")
            target_id = new_maneuver.target_id
        elif ship.current_maneuver:
            maneuver_type = ship.current_maneuver.maneuver_type.name
            throttle = ship.current_maneuver.throttle * 100
            parts.append(f"{maneuver_type} @ {throttle:.0f}%")
            target_id = ship.current_maneuver.target_id
        else:
            parts.append("DRIFT")
            target_id = None

        # Target - use maneuver target or captain's primary_target_id
        if not target_id:
            # Check captain's target (captain stores target, not ship)
            captain = self.alpha_captains.get(ship_id) or self.beta_captains.get(ship_id)
            if captain:
                target_id = captain.get_primary_target_id()

        if target_id:
            target_ship = self.simulation.get_ship(target_id)
            if target_ship:
                target_name = getattr(target_ship, 'name', target_id)
                parts.append(f"target: {target_name}")
            else:
                parts.append(f"target: {target_id}")
        else:
            parts.append("target: None")

        # Radiator status - check commands first, then current state
        radiator_cmd = None
        for cmd in commands:
            if isinstance(cmd, dict) and cmd.get('type') == 'set_radiators':
                radiator_cmd = cmd.get('extend', False)
                break

        if radiator_cmd is not None:
            parts.append("RAD:EXT" if radiator_cmd else "RAD:RET")
        elif ship.thermal_system and hasattr(ship.thermal_system, 'radiators') and ship.thermal_system.radiators:
            from ..thermal import RadiatorState
            extended = any(
                rad.state == RadiatorState.EXTENDED
                for rad in ship.thermal_system.radiators.radiators.values()
            )
            parts.append("RAD:EXT" if extended else "RAD:RET")

        return " | ".join(parts)

    def _record_captain_decision(
        self,
        ship_id: str,
        captain: LLMCaptain,
        commands: List[Any],
    ) -> None:
        """Record captain's decision to battle recording."""
        from ..simulation import Maneuver

        ship = self.simulation.get_ship(ship_id)
        if not ship or not self.recorder:
            return

        # Extract maneuver info
        maneuver_type = "DRIFT"
        throttle = 0.0
        target_id = None
        target_name = None

        for cmd in commands:
            if isinstance(cmd, Maneuver):
                maneuver_type = cmd.maneuver_type.name
                throttle = cmd.throttle
                target_id = cmd.target_id
                break

        # Get target name
        if target_id:
            target_ship = self.simulation.get_ship(target_id)
            if target_ship:
                target_name = getattr(target_ship, 'name', target_id)

        # Get radiator state
        radiators_extended = False
        for cmd in commands:
            if isinstance(cmd, dict) and cmd.get('type') == 'set_radiators':
                radiators_extended = cmd.get('extend', False)
                break
        else:
            # Fall back to current state
            if ship.thermal_system and hasattr(ship.thermal_system, 'radiators') and ship.thermal_system.radiators:
                from ..thermal import RadiatorState
                radiators_extended = any(
                    rad.state == RadiatorState.EXTENDED
                    for rad in ship.thermal_system.radiators.radiators.values()
                )

        # Get acknowledgment text from captain if available
        acknowledgment = getattr(captain, 'last_acknowledgment', None)

        self.recorder.record_captain_decision(
            timestamp=self.simulation.current_time,
            ship_id=ship_id,
            captain_name=captain.name,
            ship_name=captain.ship_name,
            maneuver_type=maneuver_type,
            throttle=throttle,
            target_id=target_id,
            target_name=target_name,
            radiators_extended=radiators_extended,
            acknowledgment=acknowledgment,
        )

    def _log_fleet_decision(self, all_commands: Dict[str, List[Any]]) -> None:
        """Log fleet decision point."""
        log_entry = {
            "checkpoint": self.checkpoint_count,
            "time": self.simulation.current_time,
            "commands_by_ship": {
                ship_id: len(cmds)
                for ship_id, cmds in all_commands.items()
            },
        }

        # Add ship states
        for ship_id in list(self.alpha_ships.keys()) + list(self.beta_ships.keys()):
            ship = self.simulation.get_ship(ship_id)
            if ship:
                log_entry[f"{ship_id}_hull"] = ship.hull_integrity

        self.decision_log.append(log_entry)

    def _is_fleet_battle_over(self) -> bool:
        """Check if fleet battle should end."""
        if self.simulation is None:
            return True

        # Check if all ships on one side are destroyed/surrendered
        alpha_active = any(
            not ship.is_destroyed and not getattr(ship, 'is_surrendered', False)
            for ship in self.alpha_ships.values()
        )
        beta_active = any(
            not ship.is_destroyed and not getattr(ship, 'is_surrendered', False)
            for ship in self.beta_ships.values()
        )

        if not alpha_active or not beta_active:
            return True

        # Check for mutual Admiral draw
        if self.alpha_admiral and self.beta_admiral:
            alpha_draw = getattr(self.alpha_admiral, 'proposed_draw', False)
            beta_draw = getattr(self.beta_admiral, 'proposed_draw', False)
            if alpha_draw and beta_draw:
                return True

        # In unlimited mode, only destruction/surrender/draw can end battle
        if self.config.unlimited_mode:
            return False

        # Time limit
        time_limit = self.fleet_config.time_limit_s if self.fleet_config else self.config.time_limit_s
        if self.simulation.current_time >= time_limit:
            return True

        return False

    def _evaluate_fleet_result(self) -> BattleResult:
        """Evaluate final fleet battle result."""
        # Count active ships
        alpha_active = sum(
            1 for ship in self.alpha_ships.values()
            if not ship.is_destroyed and not getattr(ship, 'is_surrendered', False)
        )
        beta_active = sum(
            1 for ship in self.beta_ships.values()
            if not ship.is_destroyed and not getattr(ship, 'is_surrendered', False)
        )

        # Determine outcome
        if alpha_active == 0 and beta_active == 0:
            outcome = BattleOutcome.DRAW
            winner = None
            reason = "Mutual destruction/surrender"
        elif alpha_active == 0:
            outcome = BattleOutcome.BETA_VICTORY
            winner = "beta"
            reason = "Alpha fleet eliminated"
        elif beta_active == 0:
            outcome = BattleOutcome.ALPHA_VICTORY
            winner = "alpha"
            reason = "Beta fleet eliminated"
        elif self.alpha_admiral and self.beta_admiral:
            # Check for mutual draw
            if getattr(self.alpha_admiral, 'proposed_draw', False) and getattr(self.beta_admiral, 'proposed_draw', False):
                # Resolve by fleet points
                alpha_points = sum(self._calculate_battle_points(ship) for ship in self.alpha_ships.values())
                beta_points = sum(self._calculate_battle_points(ship) for ship in self.beta_ships.values())
                if alpha_points > beta_points:
                    outcome = BattleOutcome.ALPHA_VICTORY
                    winner = "alpha"
                    reason = f"Draw accepted - Alpha wins on points ({alpha_points:.1f} vs {beta_points:.1f})"
                elif beta_points > alpha_points:
                    outcome = BattleOutcome.BETA_VICTORY
                    winner = "beta"
                    reason = f"Draw accepted - Beta wins on points ({beta_points:.1f} vs {alpha_points:.1f})"
                else:
                    outcome = BattleOutcome.DRAW
                    winner = None
                    reason = "Draw accepted - Perfect tie"
            else:
                # Time limit - evaluate by tactical advantage
                outcome, winner, reason = self._evaluate_fleet_tactical_advantage()
        else:
            # Time limit - evaluate by tactical advantage
            outcome, winner, reason = self._evaluate_fleet_tactical_advantage()

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"FLEET BATTLE RESULT: {outcome.value.upper()}")
            print(f"{'='*60}")
            print(f"Winner: {winner or 'None (Draw)'}")
            print(f"Reason: {reason}")
            print(f"Duration: {self.simulation.current_time:.0f}s")
            print(f"Checkpoints: {self.checkpoint_count}")
            print(f"Alpha ships remaining: {alpha_active}/{len(self.alpha_ships)}")
            print(f"Beta ships remaining: {beta_active}/{len(self.beta_ships)}")
            print(f"{'='*60}\n")

        # End recording and save
        if self.recorder:
            # Create a result-like object for the recorder
            class FleetRecordingResult:
                pass
            rec_result = FleetRecordingResult()
            rec_result.winner = winner
            rec_result.reason = reason
            rec_result.outcome = outcome
            rec_result.duration_s = self.simulation.current_time
            rec_result.checkpoints_used = self.checkpoint_count

            # Set fleet-specific fields on recording
            self.recorder.recording.alpha_ships_remaining = alpha_active
            self.recorder.recording.beta_ships_remaining = beta_active

            self.recorder.end_recording(rec_result, self.simulation.current_time)

            # Save to file
            filename = create_battle_filename(
                self.recorder.recording.alpha_model,
                self.recorder.recording.beta_model,
            )
            filepath = Path(self.config.recording_dir) / filename
            self.recording_file = self.recorder.save(str(filepath))

            if self.config.verbose:
                print(f"Recording saved to: {self.recording_file}")

        # Collect per-ship stats
        alpha_fleet_stats = {
            ship_id: self._collect_stats(ship)
            for ship_id, ship in self.alpha_ships.items()
        }
        beta_fleet_stats = {
            ship_id: self._collect_stats(ship)
            for ship_id, ship in self.beta_ships.items()
        }

        # For backward compatibility, use first ship for legacy stats
        first_alpha = list(self.alpha_ships.values())[0] if self.alpha_ships else None
        first_beta = list(self.beta_ships.values())[0] if self.beta_ships else None

        return BattleResult(
            outcome=outcome,
            winner=winner,
            reason=reason,
            duration_s=self.simulation.current_time,
            checkpoints_used=self.checkpoint_count,
            alpha_stats=self._collect_stats(first_alpha),
            beta_stats=self._collect_stats(first_beta),
            alpha_fleet_stats=alpha_fleet_stats,
            beta_fleet_stats=beta_fleet_stats,
            decision_log=self.decision_log,
            messages=self.fleet_communication.get_all_messages_formatted() if self.fleet_communication else [],
            is_fleet_battle=True,
            recording_file=self.recording_file,
        )

    def _is_battle_over(self) -> bool:
        """Check if battle should end."""
        if self.simulation is None:
            return True

        # Ship destroyed - always ends battle
        alpha = self.simulation.get_ship("alpha")
        beta = self.simulation.get_ship("beta")

        if alpha and alpha.is_destroyed:
            return True
        if beta and beta.is_destroyed:
            return True

        # Communication end (surrender, mutual draw) - always ends battle
        if self.communication and self.communication.is_battle_ended():
            return True

        # In unlimited mode, only destruction/surrender/draw can end battle
        if self.config.unlimited_mode:
            return False

        # Time limit (only in limited mode)
        if self.simulation.current_time >= self.config.time_limit_s:
            return True

        # Checkpoint limit (only in limited mode)
        if self.checkpoint_count >= self.config.max_checkpoints:
            return True

        return False

    def _record_sim_frame(self) -> None:
        """Record current simulation state for detailed analysis."""
        if not self.recorder or not self.simulation:
            return

        # Build ship states - iterate over ALL ships in simulation (including destroyed)
        ships = {}
        for ship_id, ship in self.simulation.ships.items():
            if ship:
                # Get current maneuver type
                maneuver_str = "MAINTAIN"
                if ship.current_maneuver and not ship.is_destroyed:
                    maneuver_str = ship.current_maneuver.maneuver_type.name

                # Get current thrust fraction
                thrust = 0.0
                if ship.current_maneuver and not ship.is_destroyed:
                    thrust = ship.current_maneuver.throttle

                # Hull integrity percentage
                hull_pct = 0.0
                if hasattr(ship, 'hull_integrity'):
                    hull_pct = ship.hull_integrity

                # Extract armor thickness per section
                armor_data = {}
                if hasattr(ship, 'armor') and ship.armor:
                    for loc, armor in ship.armor.sections.items():
                        armor_data[loc.value] = round(armor.thickness_cm, 1)

                ships[ship_id] = {
                    "position": (ship.position.x, ship.position.y, ship.position.z),
                    "velocity": (ship.velocity.x, ship.velocity.y, ship.velocity.z),
                    "forward": (ship.forward.x, ship.forward.y, ship.forward.z),
                    "thrust": thrust,
                    "maneuver": maneuver_str,
                    "is_destroyed": ship.is_destroyed,
                    "hull_pct": round(hull_pct, 1),
                    "armor": armor_data,
                }

        # Build projectile states
        projectiles = []
        for proj_flight in self.simulation.projectiles:
            proj = proj_flight.projectile
            # Check if any PD is engaging this projectile
            pd_engaged = hasattr(proj, '_pd_ablation') and proj._pd_ablation > 0

            projectiles.append({
                "id": proj_flight.projectile_id,
                "position": (proj.position.x, proj.position.y, proj.position.z),
                "velocity": (proj.velocity.x, proj.velocity.y, proj.velocity.z),
                "mass_kg": proj.mass_kg,
                "source_ship_id": proj_flight.source_ship_id,
                "target_ship_id": proj_flight.target_ship_id,
                "pd_engaged": pd_engaged,
                "pd_ablation_kg": getattr(proj, '_pd_ablation', 0.0),
            })

        # Build torpedo states
        torpedoes = []
        for torp_flight in self.simulation.torpedoes:
            torp = torp_flight.torpedo
            torpedoes.append({
                "id": torp_flight.torpedo_id,
                "position": (torp.position.x, torp.position.y, torp.position.z),
                "velocity": (torp.velocity.x, torp.velocity.y, torp.velocity.z),
                "source_ship_id": torp_flight.source_ship_id,
                "target_ship_id": getattr(torp, 'target_ship_id', None),
                "dv_remaining_kps": torp.delta_v_remaining_kps,
                "heat_absorbed_j": torp_flight.heat_absorbed_j,
                "is_disabled": torp_flight.is_disabled,
            })

        self.recorder.record_sim_frame(
            timestamp=self.simulation.current_time,
            ships=ships,
            projectiles=projectiles,
            torpedoes=torpedoes,
        )

    def _print_status(self) -> None:
        """Print current battle status."""
        alpha = self.simulation.get_ship("alpha")
        beta = self.simulation.get_ship("beta")

        if alpha and beta:
            dist = (alpha.position - beta.position).magnitude / 1000
            print(f"  Distance: {dist:.0f} km")

            # Get armor status for all sections
            from ..combat import HitLocation
            def get_armor(ship, loc):
                sec = ship.armor.get_section(loc) if ship.armor else None
                return sec.thickness_cm if sec else 0

            alpha_nose = get_armor(alpha, HitLocation.NOSE)
            alpha_lat = get_armor(alpha, HitLocation.LATERAL)
            alpha_tail = get_armor(alpha, HitLocation.TAIL)
            beta_nose = get_armor(beta, HitLocation.NOSE)
            beta_lat = get_armor(beta, HitLocation.LATERAL)
            beta_tail = get_armor(beta, HitLocation.TAIL)

            print(f"  Alpha: {alpha.hull_integrity:.0f}% hull, {alpha.thermal_system.heat_percent:.0f}% heat, armor: N{alpha_nose:.0f}/L{alpha_lat:.0f}/T{alpha_tail:.0f}cm, shots: {alpha.shots_fired}, hits: {alpha.hits_scored}")
            print(f"  Beta: {beta.hull_integrity:.0f}% hull, {beta.thermal_system.heat_percent:.0f}% heat, armor: N{beta_nose:.0f}/L{beta_lat:.0f}/T{beta_tail:.0f}cm, shots: {beta.shots_fired}, hits: {beta.hits_scored}")

    def _log_decision(
        self,
        alpha_commands: List[Any],
        beta_commands: List[Any],
    ) -> None:
        """Log decision point."""
        alpha = self.simulation.get_ship("alpha")
        beta = self.simulation.get_ship("beta")

        self.decision_log.append({
            "checkpoint": self.checkpoint_count,
            "time": self.simulation.current_time,
            "alpha_commands": len(alpha_commands),
            "beta_commands": len(beta_commands),
            "alpha_hull": alpha.hull_integrity if alpha else 0,
            "beta_hull": beta.hull_integrity if beta else 0,
        })

    def _calculate_battle_points(self, ship: Any) -> float:
        """
        Calculate battle points for a ship.

        Used for determining winner when both captains agree to draw.
        Points are awarded for:
        - Damage dealt (1 point per GJ)
        - Hull integrity bonus (up to 50 points)
        - Accuracy bonus (up to 20 points)
        - Delta-v remaining bonus (up to 30 points)

        Args:
            ship: Ship to calculate points for

        Returns:
            Total battle points
        """
        if ship is None:
            return 0.0

        points = 0.0

        # Damage dealt (1 point per GJ)
        points += ship.damage_dealt_gj

        # Hull integrity bonus (up to 50 points)
        points += ship.hull_integrity * 0.5  # 100% hull = 50 points

        # Accuracy bonus (up to 20 points)
        if ship.shots_fired > 0:
            accuracy = ship.hits_scored / ship.shots_fired
            points += accuracy * 20

        # Delta-v remaining bonus (up to 30 points, assuming 500 km/s budget)
        remaining_dv_pct = min(1.0, ship.remaining_delta_v_kps / 500.0)
        points += remaining_dv_pct * 30

        return points

    def _resolve_draw(self, alpha: Any, beta: Any) -> tuple:
        """
        Resolve a mutual draw by calculating battle points.

        The game ends (both captains agreed to draw), but a winner is
        still determined based on battle performance.

        Args:
            alpha: Alpha ship
            beta: Beta ship

        Returns:
            Tuple of (outcome, winner, reason)
        """
        alpha_points = self._calculate_battle_points(alpha)
        beta_points = self._calculate_battle_points(beta)

        # Print points breakdown if verbose
        if self.config.verbose:
            print(f"\n[DRAW RESOLUTION]")
            print(f"  Alpha points: {alpha_points:.1f}")
            print(f"  Beta points: {beta_points:.1f}")

        if alpha_points > beta_points:
            return (
                BattleOutcome.ALPHA_VICTORY,
                "alpha",
                f"Draw accepted - Alpha wins on points ({alpha_points:.1f} vs {beta_points:.1f})"
            )
        elif beta_points > alpha_points:
            return (
                BattleOutcome.BETA_VICTORY,
                "beta",
                f"Draw accepted - Beta wins on points ({beta_points:.1f} vs {alpha_points:.1f})"
            )
        else:
            return (
                BattleOutcome.DRAW,
                None,
                f"Draw accepted - Perfect tie ({alpha_points:.1f} points each)"
            )

    def _evaluate_fleet_tactical_advantage(self) -> tuple:
        """
        Evaluate fleet battle by tactical advantage when time limit reached.

        Calculates a score for each fleet based on:
        - Ships destroyed/surrendered: Major penalty for losses
        - Hull integrity: Remaining health of surviving ships
        - Damage dealt: Combat effectiveness
        - Accuracy: Hit rate

        Returns:
            Tuple of (outcome, winner, reason)
        """
        # Calculate fleet scores
        alpha_score = 0.0
        beta_score = 0.0

        # Score for surviving ships (ship value based on class)
        ship_values = {
            "corvette": 10,
            "frigate": 15,
            "destroyer": 20,
            "cruiser": 30,
            "battlecruiser": 40,
            "battleship": 50,
            "dreadnought": 70,
        }

        # Alpha fleet scoring
        for ship in self.alpha_ships.values():
            ship_type = getattr(ship, 'ship_type', 'destroyer').lower()
            base_value = ship_values.get(ship_type, 20)

            if ship.is_destroyed:
                # Destroyed ships count against you
                beta_score += base_value * 1.5  # Enemy gets points for kills
            elif getattr(ship, 'is_surrendered', False):
                beta_score += base_value * 0.75  # Partial credit for surrenders
            else:
                # Surviving ships contribute to score
                hull_pct = ship.hull_integrity / 100.0
                alpha_score += base_value * hull_pct  # Hull remaining
                alpha_score += ship.damage_dealt_gj * 0.1  # Damage dealt
                if ship.shots_fired > 0:
                    accuracy = ship.hits_scored / ship.shots_fired
                    alpha_score += accuracy * 10  # Accuracy bonus

        # Beta fleet scoring
        for ship in self.beta_ships.values():
            ship_type = getattr(ship, 'ship_type', 'destroyer').lower()
            base_value = ship_values.get(ship_type, 20)

            if ship.is_destroyed:
                alpha_score += base_value * 1.5
            elif getattr(ship, 'is_surrendered', False):
                alpha_score += base_value * 0.75
            else:
                hull_pct = ship.hull_integrity / 100.0
                beta_score += base_value * hull_pct
                beta_score += ship.damage_dealt_gj * 0.1
                if ship.shots_fired > 0:
                    accuracy = ship.hits_scored / ship.shots_fired
                    beta_score += accuracy * 10

        # Print breakdown if verbose
        if self.config.verbose:
            print(f"\n[TACTICAL ADVANTAGE EVALUATION]")
            print(f"  Alpha fleet score: {alpha_score:.1f}")
            print(f"  Beta fleet score: {beta_score:.1f}")

        # Margin for draw (5% of higher score or minimum 5 points)
        margin = max(5.0, max(alpha_score, beta_score) * 0.05)

        if abs(alpha_score - beta_score) < margin:
            return (
                BattleOutcome.DRAW,
                None,
                f"Too close to call ({alpha_score:.1f} vs {beta_score:.1f})"
            )
        elif alpha_score > beta_score:
            return (
                BattleOutcome.ALPHA_VICTORY,
                "alpha",
                f"Alpha tactical advantage ({alpha_score:.1f} vs {beta_score:.1f})"
            )
        else:
            return (
                BattleOutcome.BETA_VICTORY,
                "beta",
                f"Beta tactical advantage ({beta_score:.1f} vs {alpha_score:.1f})"
            )

    def _evaluate_result(self) -> BattleResult:
        """Evaluate final battle result."""
        alpha = self.simulation.get_ship("alpha")
        beta = self.simulation.get_ship("beta")

        # Check for mutual draw first - resolve by points
        mutual_draw = self.communication.has_mutual_draw() if self.communication else False
        if mutual_draw:
            outcome, winner, reason = self._resolve_draw(alpha, beta)
        else:
            # Determine if at time/checkpoint limit (never true in unlimited mode)
            if self.config.unlimited_mode:
                at_limit = False
            else:
                at_limit = (
                    self.simulation.current_time >= self.config.time_limit_s or
                    self.checkpoint_count >= self.config.max_checkpoints
                )

            outcome, winner, reason = self.evaluator.evaluate(
                alpha=alpha,
                beta=beta,
                alpha_surrendered=self.communication.alpha_surrendered if self.communication else False,
                beta_surrendered=self.communication.beta_surrendered if self.communication else False,
                mutual_draw=False,  # Already handled above
                at_time_limit=at_limit,
            )

        # End recording and save
        if self.recorder:
            # Create a result-like object for the recorder
            class RecordingResult:
                pass
            rec_result = RecordingResult()
            rec_result.winner = winner
            rec_result.reason = reason
            rec_result.outcome = outcome
            rec_result.duration_s = self.simulation.current_time
            rec_result.checkpoints_used = self.checkpoint_count

            self.recorder.end_recording(rec_result, self.simulation.current_time)

            # Save to file
            filename = create_battle_filename(
                self.alpha_config.model,
                self.beta_config.model,
            )
            filepath = Path(self.config.recording_dir) / filename
            self.recording_file = self.recorder.save(str(filepath))

            if self.config.verbose:
                print(f"Recording saved to: {self.recording_file}")

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"BATTLE RESULT: {outcome.value.upper()}")
            print(f"{'='*60}")
            print(f"Winner: {winner or 'None (Draw)'}")
            print(f"Reason: {reason}")
            print(f"Duration: {self.simulation.current_time:.0f}s")
            print(f"Checkpoints: {self.checkpoint_count}")
            print(f"{'='*60}\n")

        return BattleResult(
            outcome=outcome,
            winner=winner,
            reason=reason,
            duration_s=self.simulation.current_time,
            checkpoints_used=self.checkpoint_count,
            alpha_stats=self._collect_stats(alpha),
            beta_stats=self._collect_stats(beta),
            decision_log=self.decision_log,
            messages=self.communication.get_all_messages_formatted() if self.communication else [],
            recording_file=self.recording_file,
        )

    def _collect_stats(self, ship: Any) -> Dict[str, Any]:
        """Collect final statistics for a ship."""
        if ship is None:
            return {}

        return {
            "hull_integrity": ship.hull_integrity,
            "is_destroyed": ship.is_destroyed,
            "shots_fired": ship.shots_fired,
            "hits_scored": ship.hits_scored,
            "damage_dealt": ship.damage_dealt_gj,
            "damage_taken": ship.damage_taken_gj,
            "delta_v_remaining": ship.remaining_delta_v_kps,
        }

    def _build_ship_state_dict(self, ship: Any) -> Dict[str, Any]:
        """Build a comprehensive ship state dictionary for recording."""
        from ..combat import HitLocation
        from ..thermal import RadiatorState

        state = {
            "ship_id": ship.ship_id,
            "position": (ship.position.x, ship.position.y, ship.position.z),
            "velocity": (ship.velocity.x, ship.velocity.y, ship.velocity.z),
            "forward": (ship.forward.x, ship.forward.y, ship.forward.z),
            "hull_integrity": ship.hull_integrity,
            "is_destroyed": ship.is_destroyed,
            "shots_fired": ship.shots_fired,
            "hits_scored": ship.hits_scored,
            "damage_dealt_gj": ship.damage_dealt_gj,
            "damage_taken_gj": ship.damage_taken_gj,
            "delta_v_remaining_kps": ship.remaining_delta_v_kps,
        }

        # Thermal
        if ship.thermal_system:
            state["heat_percent"] = ship.thermal_system.heat_percent
            state["heatsink_energy_gj"] = ship.thermal_system.heatsink.current_heat_gj
            state["heatsink_capacity_gj"] = ship.thermal_system.heatsink.capacity_gj

            # Radiator state
            if ship.thermal_system.radiators:
                extended = any(
                    rad.state == RadiatorState.EXTENDED
                    for rad in ship.thermal_system.radiators.radiators.values()
                )
                state["radiators_extended"] = extended
            else:
                state["radiators_extended"] = False
        else:
            state["heat_percent"] = 0
            state["radiators_extended"] = False

        # Armor
        if ship.armor:
            armor_state = {}
            for section_name in ["nose", "lateral", "tail"]:
                try:
                    section = ship.armor.get_section(HitLocation[section_name.upper()])
                    if section:
                        armor_state[section_name] = {
                            "thickness_cm": section.thickness_cm,
                            "original_cm": section.original_thickness_cm if hasattr(section, 'original_thickness_cm') else section.thickness_cm,
                        }
                except (KeyError, AttributeError):
                    pass
            state["armor"] = armor_state

        # Weapons ammo (if available)
        if hasattr(ship, 'weapons') and ship.weapons:
            weapons_state = {}
            for slot, weapon in ship.weapons.items():
                weapon_info = {
                    "type": type(weapon).__name__,
                }
                if hasattr(weapon, 'ammo_remaining'):
                    weapon_info["ammo_remaining"] = weapon.ammo_remaining
                if hasattr(weapon, 'ammo_capacity'):
                    weapon_info["ammo_capacity"] = weapon.ammo_capacity
                if hasattr(weapon, 'cooldown_remaining'):
                    weapon_info["cooldown_s"] = weapon.cooldown_remaining
                weapons_state[slot] = weapon_info
            state["weapons"] = weapons_state

        # Module health
        if hasattr(ship, 'module_layout') and ship.module_layout:
            modules_state = {}
            # Use get_all_modules() method if available, else try .modules attribute
            if hasattr(ship.module_layout, 'get_all_modules'):
                modules = ship.module_layout.get_all_modules()
            elif hasattr(ship.module_layout, 'modules'):
                modules = ship.module_layout.modules
            else:
                modules = []
            for module in modules:
                modules_state[module.name] = {
                    "health_percent": module.health_percent,
                    "is_destroyed": module.is_destroyed,
                    "is_functional": module.is_functional,
                    "is_critical": module.is_critical,
                }
            state["modules"] = modules_state

        return state

    def _handle_simulation_event(self, event: Any) -> None:
        """
        Handle simulation events and record them.

        This is called by the simulation for every event (shots, hits, damage, etc.)
        """
        if not self.recorder:
            return

        from ..simulation import SimulationEventType

        event_type = event.event_type
        timestamp = event.timestamp
        ship_id = event.ship_id
        target_id = event.target_id
        data = event.data

        # Record projectile launches (shots fired)
        if event_type == SimulationEventType.PROJECTILE_LAUNCHED:
            self.recorder.record_shot_fired(
                timestamp=timestamp,
                shooter_id=ship_id,
                target_id=target_id or "unknown",
                weapon_slot=data.get("weapon_slot", "unknown"),
                weapon_name=data.get("weapon_name", "unknown"),
                hit_probability=data.get("hit_probability", 0),
                distance_km=data.get("distance_km", 0),
                eta_s=data.get("eta_s", 0),
                projectile_energy_gj=data.get("kinetic_energy_gj", 0),
                muzzle_velocity_kps=data.get("muzzle_velocity_kps", 0),
            )

        # Record projectile impacts (hits)
        # Note: event.ship_id is the shooter, event.target_id is the target
        elif event_type == SimulationEventType.PROJECTILE_IMPACT:
            self.recorder.record_hit(
                timestamp=timestamp,
                shooter_id=ship_id or "unknown",
                target_id=target_id or "unknown",
                weapon_slot=data.get("weapon_slot", "unknown"),
                hit_location=data.get("hit_location", "unknown"),
                impact_angle_deg=data.get("impact_angle_deg", 0),
                kinetic_energy_gj=data.get("kinetic_energy_gj", 0),
                armor_ablation_cm=data.get("armor_ablation_cm", 0),
                armor_remaining_cm=data.get("armor_remaining_cm", 0),
                damage_to_hull_gj=data.get("damage_to_hull_gj", 0),
                penetrated=data.get("penetrated", False),
                critical_hit=data.get("critical_hit", False),
                flight_time_s=data.get("flight_time_s", 0),
                projectile_id=data.get("projectile_id"),
                impact_position=data.get("impact_position"),
            )

            # Record shot for captain learning
            source_id = data.get("source_ship_id", "unknown")
            self._record_captain_shot(
                source_id=source_id,
                weapon_slot=data.get("weapon_slot", "unknown"),
                result="HIT",
                damage_gj=data.get("kinetic_energy_gj", 0),
            )

        # Record misses
        elif event_type == SimulationEventType.PROJECTILE_MISS:
            self.recorder.record_miss(
                timestamp=timestamp,
                shooter_id=data.get("source_ship_id", ship_id or "unknown"),
                target_id=target_id or "unknown",
                weapon_slot=data.get("weapon_slot", "unknown"),
                hit_probability=data.get("hit_probability", 0),
                distance_km=data.get("closest_approach_km", 0),
                flight_time_s=data.get("flight_time_s", 0),
            )

            # Record shot for captain learning
            source_id = data.get("source_ship_id", ship_id or "unknown")
            self._record_captain_shot(
                source_id=source_id,
                weapon_slot=data.get("weapon_slot", "unknown"),
                result="MISS",
                damage_gj=0.0,
            )

        # Record armor penetration
        elif event_type == SimulationEventType.ARMOR_PENETRATED:
            self.recorder.record_armor_damage(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                location=data.get("hit_location", "unknown"),
                ablation_cm=data.get("armor_ablation_cm", 0),
                remaining_cm=data.get("armor_remaining_cm", 0),
                chipping_fraction=data.get("chipping_fraction", 0),
            )

        # Record radiator changes
        elif event_type == SimulationEventType.RADIATOR_EXTENDED:
            self.recorder.record_radiator_change(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                extended=True,
            )
        elif event_type == SimulationEventType.RADIATOR_RETRACTED:
            self.recorder.record_radiator_change(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                extended=False,
            )

        # Record thermal warnings
        elif event_type == SimulationEventType.THERMAL_WARNING:
            self.recorder.record_thermal_warning(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                heat_percent=data.get("heat_percent", 0),
                is_critical=False,
            )
        elif event_type == SimulationEventType.THERMAL_CRITICAL:
            self.recorder.record_thermal_warning(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                heat_percent=data.get("heat_percent", 0),
                is_critical=True,
            )

        # Record maneuvers
        elif event_type == SimulationEventType.MANEUVER_STARTED:
            self.recorder.record_maneuver(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                maneuver_type=data.get("maneuver_type", "unknown"),
                throttle=data.get("throttle", 1.0),
                target_id=data.get("target_id"),
            )

        # Record module damage and destruction
        elif event_type == SimulationEventType.MODULE_DAMAGED:
            self.recorder.record_module_damaged(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                module_name=data.get("module_name", "unknown"),
                damage_gj=data.get("damage_gj", 0),
                destroyed=data.get("destroyed", False),
            )

        elif event_type == SimulationEventType.MODULE_DESTROYED:
            self.recorder.record_module_destroyed(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                module_name=data.get("module_name", "unknown"),
            )

        # Record point defense events
        elif event_type == SimulationEventType.PD_ENGAGED:
            self.recorder.record_pd_fired(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                turret_name=data.get("turret", "unknown"),
                target_type=data.get("target_type", "unknown"),
                target_id=data.get("target_id", "unknown"),
                distance_km=data.get("distance_km", 0),
                mass_ablated_kg=data.get("mass_ablated_kg", 0),
                total_ablated_kg=data.get("total_ablated_kg", 0),
                energy_delivered_j=data.get("energy_delivered_j", 0),
            )

        elif event_type == SimulationEventType.PD_SLUG_DAMAGED:
            self.recorder.record_pd_slug_damaged(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                projectile_id=data.get("projectile_id", "unknown"),
                remaining_mass_kg=data.get("remaining_mass_kg", 0),
            )

        elif event_type == SimulationEventType.PD_SLUG_DESTROYED:
            self.recorder.record_pd_slug_destroyed(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                projectile_id=data.get("projectile_id", "unknown"),
                source_ship_id=data.get("source_ship", "unknown"),
            )

        elif event_type == SimulationEventType.PD_TORPEDO_DISABLED:
            self.recorder.record_pd_torpedo_disabled(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                torpedo_id=data.get("torpedo_id", "unknown"),
                source_ship_id=data.get("source_ship", "unknown"),
            )

        elif event_type == SimulationEventType.PD_TORPEDO_DESTROYED:
            self.recorder.record_pd_torpedo_destroyed(
                timestamp=timestamp,
                ship_id=ship_id or "unknown",
                torpedo_id=data.get("torpedo_id", "unknown"),
                source_ship_id=data.get("source_ship", "unknown"),
                total_heat_absorbed_j=data.get("total_heat_absorbed_j", 0),
            )


    def _record_captain_shot(
        self,
        source_id: str,
        weapon_slot: str,
        result: str,
        damage_gj: float,
    ) -> None:
        """Record a shot for captain learning feedback.

        Args:
            source_id: Ship that fired ("alpha" or "beta")
            weapon_slot: Which weapon fired
            result: "HIT" or "MISS"
            damage_gj: Damage dealt (for hits)
        """
        if not self.simulation:
            return

        # Get the captain who fired
        captain = self.alpha_captain if source_id == "alpha" else self.beta_captain
        if not captain:
            return

        # Get ships for distance/velocity calculation
        alpha = self.simulation.get_ship("alpha")
        beta = self.simulation.get_ship("beta")
        if not alpha or not beta:
            return

        # Calculate distance
        distance_m = (alpha.position - beta.position).magnitude
        distance_km = distance_m / 1000

        # Calculate relative velocity (closing rate)
        rel_vel = beta.velocity - alpha.velocity if source_id == "alpha" else alpha.velocity - beta.velocity
        rel_pos = beta.position - alpha.position if source_id == "alpha" else alpha.position - beta.position
        if distance_m > 0:
            # Positive = closing, negative = separating (from shooter's perspective)
            closing_kps = -rel_pos.normalized().dot(rel_vel) / 1000
        else:
            closing_kps = 0.0

        # Get weapon name instead of slot
        weapon_name = weapon_slot
        shooter = alpha if source_id == "alpha" else beta
        if hasattr(shooter, 'weapons') and weapon_slot in shooter.weapons:
            ws = shooter.weapons[weapon_slot]
            if hasattr(ws, 'weapon') and hasattr(ws.weapon, 'name'):
                weapon_name = ws.weapon.name

        captain.record_shot(
            weapon=weapon_name,
            distance_km=distance_km,
            rel_velocity_kps=closing_kps,
            result=result,
            damage_gj=damage_gj,
        )


def load_fleet_data(path: Optional[str] = None) -> Dict[str, Any]:
    """Load fleet data from JSON file."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "fleet_ships.json"
    else:
        path = Path(path)

    with open(path) as f:
        return json.load(f)
