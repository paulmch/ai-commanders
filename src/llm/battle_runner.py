"""
LLM Battle Runner - Orchestrates battles between LLM-controlled captains.

Handles simulation setup, checkpoint timing, and victory evaluation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .client import CaptainClient
from .captain import LLMCaptain, LLMCaptainConfig
from .communication import CommunicationChannel, MessageType
from .victory import VictoryEvaluator, BattleOutcome
from .prompts import CaptainPersonality
from .battle_recorder import BattleRecorder, create_battle_filename


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

    # Recording file path (if recorded)
    recording_file: Optional[str] = None


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
        self.recorder: Optional[BattleRecorder] = None

        self.checkpoint_count = 0
        self.decision_log: List[Dict[str, Any]] = []
        self.recording_file: Optional[str] = None

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

    def run_battle(self, fleet_data: Dict[str, Any]) -> BattleResult:
        """
        Run a complete battle with LLM-controlled captains.

        Args:
            fleet_data: Ship specifications

        Returns:
            BattleResult with outcome and statistics
        """
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
            alpha_msg = self.alpha_captain.get_pending_message()
            beta_msg = self.beta_captain.get_pending_message()

            if alpha_msg:
                self.communication.queue_message(
                    "alpha", alpha_msg, self.simulation.current_time
                )
                if self.recorder:
                    self.recorder.record_message(
                        timestamp=self.simulation.current_time,
                        sender_id="alpha",
                        sender_name=self.alpha_config.name,
                        ship_name=self.alpha_config.ship_name,
                        message=alpha_msg,
                    )
                if self.config.verbose:
                    print(f"  [{self.alpha_config.ship_name}] {self.alpha_config.name}: \"{alpha_msg}\"")

            if beta_msg:
                self.communication.queue_message(
                    "beta", beta_msg, self.simulation.current_time
                )
                if self.recorder:
                    self.recorder.record_message(
                        timestamp=self.simulation.current_time,
                        sender_id="beta",
                        sender_name=self.beta_config.name,
                        ship_name=self.beta_config.ship_name,
                        message=beta_msg,
                    )
                if self.config.verbose:
                    print(f"  [{self.beta_config.ship_name}] {self.beta_config.name}: \"{beta_msg}\"")

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
            if self.alpha_captain.has_proposed_draw:
                self.communication.alpha_proposed_draw = True
                if self.recorder:
                    self.recorder.record_draw_proposal(
                        timestamp=self.simulation.current_time,
                        ship_id="alpha",
                        captain_name=self.alpha_config.name,
                    )
                if self.config.verbose:
                    print(f"  [DRAW PROPOSED] {self.alpha_config.name} proposes draw")
            if self.beta_captain.has_proposed_draw:
                self.communication.beta_proposed_draw = True
                if self.recorder:
                    self.recorder.record_draw_proposal(
                        timestamp=self.simulation.current_time,
                        ship_id="beta",
                        captain_name=self.beta_config.name,
                    )
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

            # Phase 6: Check checkpoint limit (skip in unlimited mode)
            if not self.config.unlimited_mode:
                if self.checkpoint_count >= self.config.max_checkpoints:
                    if self.config.verbose:
                        print(f"\n=== CHECKPOINT LIMIT REACHED ===")
                    break

        return self._evaluate_result()

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

        # Build ship states
        ships = {}
        for ship_id in ["alpha", "beta"]:
            ship = self.simulation.get_ship(ship_id)
            if ship:
                # Get current maneuver type
                maneuver_str = "MAINTAIN"
                if ship.current_maneuver:
                    maneuver_str = ship.current_maneuver.maneuver_type.name

                # Get current thrust fraction
                thrust = 0.0
                if ship.current_maneuver:
                    thrust = ship.current_maneuver.throttle

                ships[ship_id] = {
                    "position": (ship.position.x, ship.position.y, ship.position.z),
                    "velocity": (ship.velocity.x, ship.velocity.y, ship.velocity.z),
                    "forward": (ship.forward.x, ship.forward.y, ship.forward.z),
                    "thrust": thrust,
                    "maneuver": maneuver_str,
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

    def _evaluate_result(self) -> BattleResult:
        """Evaluate final battle result."""
        alpha = self.simulation.get_ship("alpha")
        beta = self.simulation.get_ship("beta")

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
            mutual_draw=self.communication.has_mutual_draw() if self.communication else False,
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
                hit_probability=data.get("hit_probability", 0),
                distance_km=data.get("distance_m", 0) / 1000,
                projectile_energy_gj=data.get("projectile_energy_gj", 0),
                muzzle_velocity_kps=data.get("muzzle_velocity_kps", 0),
            )

        # Record projectile impacts (hits)
        elif event_type == SimulationEventType.PROJECTILE_IMPACT:
            self.recorder.record_hit(
                timestamp=timestamp,
                shooter_id=data.get("source_ship_id", "unknown"),
                target_id=ship_id or "unknown",
                weapon_slot=data.get("weapon_slot", "unknown"),
                hit_location=data.get("hit_location", "unknown"),
                impact_angle_deg=data.get("impact_angle_deg", 0),
                kinetic_energy_gj=data.get("kinetic_energy_gj", 0),
                armor_ablation_cm=data.get("armor_ablation_cm", 0),
                armor_remaining_cm=data.get("armor_remaining_cm", 0),
                damage_to_hull_gj=data.get("damage_to_hull_gj", 0),
                penetrated=data.get("penetrated", False),
                critical_hit=data.get("critical_hit", False),
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
