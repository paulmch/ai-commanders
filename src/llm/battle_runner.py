"""
LLM Battle Runner - Orchestrates battles between LLM-controlled captains.

Handles simulation setup, checkpoint timing, and victory evaluation.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

from .client import CaptainClient
from .captain import LLMCaptain, LLMCaptainConfig
from .communication import CommunicationChannel, MessageType
from .victory import VictoryEvaluator, BattleOutcome
from .prompts import CaptainPersonality


@dataclass
class BattleConfig:
    """Configuration for an LLM battle."""
    # Scenario
    initial_distance_km: float = 500.0  # Start closer for faster engagement
    initial_offset_km: float = 1.0  # Y-axis offset
    time_limit_s: float = 1200.0  # 20 minutes
    decision_interval_s: float = 30.0
    max_checkpoints: int = 40  # More passes

    # Ship types
    alpha_ship_type: str = "destroyer"
    beta_ship_type: str = "destroyer"

    # Verbose output
    verbose: bool = True


@dataclass
class BattleResult:
    """Result of an LLM battle."""
    outcome: BattleOutcome
    winner: Optional[str]
    reason: str
    duration_s: float
    checkpoints_used: int

    # Per-side stats
    alpha_stats: Dict[str, Any]
    beta_stats: Dict[str, Any]

    # Logs
    decision_log: List[Dict[str, Any]]
    messages: List[str]


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
    ):
        self.config = config
        self.alpha_config = alpha_config
        self.beta_config = beta_config
        self.client = client

        self.simulation = None
        self.alpha_captain: Optional[LLMCaptain] = None
        self.beta_captain: Optional[LLMCaptain] = None
        self.communication: Optional[CommunicationChannel] = None
        self.evaluator = VictoryEvaluator()

        self.checkpoint_count = 0
        self.decision_log: List[Dict[str, Any]] = []

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

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"LLM BATTLE: {self.alpha_config.ship_name} vs {self.beta_config.ship_name}")
            print(f"{'='*60}")
            print(f"Distance: {self.config.initial_distance_km} km")
            print(f"Alpha: {self.alpha_config.name} ({self.alpha_config.personality.value})")
            print(f"Beta: {self.beta_config.name} ({self.beta_config.personality.value})")
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

    def run_battle(self, fleet_data: Dict[str, Any]) -> BattleResult:
        """
        Run a complete battle with LLM-controlled captains.

        Args:
            fleet_data: Ship specifications

        Returns:
            BattleResult with outcome and statistics
        """
        self.setup_battle(fleet_data)

        while not self._is_battle_over():
            # === SIMULATION PHASE ===
            # Run for decision_interval seconds
            steps = int(self.config.decision_interval_s)
            for _ in range(steps):
                self.simulation.step()
                if self._is_battle_over():
                    break

            if self._is_battle_over():
                break

            # === CHECKPOINT ===
            self.checkpoint_count += 1

            if self.config.verbose:
                print(f"\n=== CHECKPOINT {self.checkpoint_count} at T+{self.simulation.current_time:.0f}s ===")
                self._print_status()

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
            alpha_msg = self.alpha_captain.get_pending_message()
            beta_msg = self.beta_captain.get_pending_message()

            if alpha_msg:
                self.communication.queue_message(
                    "alpha", alpha_msg, self.simulation.current_time
                )
                if self.config.verbose:
                    print(f"  [{self.alpha_config.ship_name}] {self.alpha_config.name}: \"{alpha_msg}\"")

            if beta_msg:
                self.communication.queue_message(
                    "beta", beta_msg, self.simulation.current_time
                )
                if self.config.verbose:
                    print(f"  [{self.beta_config.ship_name}] {self.beta_config.name}: \"{beta_msg}\"")

            # Phase 4: Check surrender/draw
            if self.alpha_captain.has_surrendered:
                self.communication.alpha_surrendered = True
            if self.beta_captain.has_surrendered:
                self.communication.beta_surrendered = True
            if self.alpha_captain.has_proposed_draw:
                self.communication.alpha_proposed_draw = True
                if self.config.verbose:
                    print(f"  [DRAW PROPOSED] {self.alpha_config.name} proposes draw")
            if self.beta_captain.has_proposed_draw:
                self.communication.beta_proposed_draw = True
                if self.config.verbose:
                    print(f"  [DRAW PROPOSED] {self.beta_config.name} proposes draw")

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

            # Phase 6: Check checkpoint limit
            if self.checkpoint_count >= self.config.max_checkpoints:
                if self.config.verbose:
                    print(f"\n=== CHECKPOINT LIMIT REACHED ===")
                break

        return self._evaluate_result()

    def _is_battle_over(self) -> bool:
        """Check if battle should end."""
        if self.simulation is None:
            return True

        # Time limit
        if self.simulation.current_time >= self.config.time_limit_s:
            return True

        # Ship destroyed
        alpha = self.simulation.get_ship("alpha")
        beta = self.simulation.get_ship("beta")

        if alpha and alpha.is_destroyed:
            return True
        if beta and beta.is_destroyed:
            return True

        # Communication end
        if self.communication and self.communication.is_battle_ended():
            return True

        return False

    def _print_status(self) -> None:
        """Print current battle status."""
        alpha = self.simulation.get_ship("alpha")
        beta = self.simulation.get_ship("beta")

        if alpha and beta:
            dist = (alpha.position - beta.position).magnitude / 1000
            print(f"  Distance: {dist:.0f} km")

            # Get armor status
            from ..combat import HitLocation
            alpha_nose_sec = alpha.armor.get_section(HitLocation.NOSE) if alpha.armor else None
            alpha_nose = alpha_nose_sec.thickness_cm if alpha_nose_sec else 0
            beta_nose_sec = beta.armor.get_section(HitLocation.NOSE) if beta.armor else None
            beta_nose = beta_nose_sec.thickness_cm if beta_nose_sec else 0

            print(f"  Alpha: {alpha.hull_integrity:.0f}% hull, {alpha.thermal_system.heat_percent:.0f}% heat, nose armor: {alpha_nose:.1f}cm, shots: {alpha.shots_fired}, hits: {alpha.hits_scored}")
            print(f"  Beta: {beta.hull_integrity:.0f}% hull, {beta.thermal_system.heat_percent:.0f}% heat, nose armor: {beta_nose:.1f}cm, shots: {beta.shots_fired}, hits: {beta.hits_scored}")

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

    def _evaluate_result(self) -> BattleResult:
        """Evaluate final battle result."""
        alpha = self.simulation.get_ship("alpha")
        beta = self.simulation.get_ship("beta")

        # Determine if at time/checkpoint limit
        at_limit = (
            self.simulation.current_time >= self.config.time_limit_s or
            self.checkpoint_count >= self.config.max_checkpoints
        )

        outcome, winner, reason = self.evaluator.evaluate(
            alpha=alpha,
            beta=beta,
            alpha_surrendered=self.communication.alpha_surrendered if self.communication else False,
            beta_surrendered=self.communication.beta_surrendered if self.communication else False,
            mutual_draw=self.communication.has_mutual_draw() if self.communication else False,
            at_time_limit=at_limit,
        )

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


def load_fleet_data(path: Optional[str] = None) -> Dict[str, Any]:
    """Load fleet data from JSON file."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "fleet_ships.json"
    else:
        path = Path(path)

    with open(path) as f:
        return json.load(f)
