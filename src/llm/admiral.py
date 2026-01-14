"""
LLM Admiral - Fleet commander that issues orders to ship captains.

The Admiral:
- Sees dual temporal snapshots (T-15s and T=0) for trajectory analysis
- Has full visibility into friendly ship status and capabilities
- Issues text orders to captains
- Can negotiate with enemy Admiral (immediate)
- Controls draw proposals for the fleet
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from .fleet_config import AdmiralConfig
from .admiral_tools import get_admiral_tools
from .communication import CaptainMessage

if TYPE_CHECKING:
    from .client import CaptainClient
    from .captain import LLMCaptain


@dataclass
class AdmiralOrder:
    """A single order from Admiral to a specific ship."""
    target_ship_id: str  # Ship ID receiving this order
    target_ship_name: str  # Ship name for display
    order_text: str  # Free-form text instruction
    priority: str = "NORMAL"  # CRITICAL, HIGH, NORMAL, LOW
    suggested_target: Optional[str] = None  # Enemy ship name to focus on


@dataclass
class AdmiralDecision:
    """Complete decision package from an Admiral."""
    timestamp: float
    fleet_orders: List[AdmiralOrder] = field(default_factory=list)
    fleet_directive: str = ""  # Overall strategic intent
    proposed_draw: bool = False
    accepted_draw: bool = False
    rejected_draw: bool = False
    message_to_enemy_admiral: Optional[str] = None
    reasoning: str = ""  # Admiral's tactical reasoning (for logs)


@dataclass
class FriendlyShipSnapshot:
    """Full status snapshot for a friendly ship."""
    ship_id: str
    ship_name: str
    ship_type: str
    captain_name: str

    # Position/velocity (in km and km/s)
    position_km: Dict[str, float]
    velocity_kps: float
    velocity_vector: Dict[str, float]

    # Combat status
    hull_integrity: float
    delta_v_remaining: float
    heat_percent: float

    # Ship capabilities (from fleet data)
    max_acceleration_g: float
    max_delta_v: float
    weapons_summary: str  # e.g., "1x Spinal (4.3 GJ), 2x Coilgun (0.7 GJ)"

    # Weapon status
    weapons_ready: List[str]
    weapons_cooling: List[str]
    weapons_destroyed: List[str]  # Destroyed weapon systems

    # Current orders/state
    current_maneuver: str
    current_target: Optional[str]
    radiators_extended: bool

    # Targeting info
    targeted_by: List[str]  # Enemy ship names targeting this ship


@dataclass
class EnemyShipSnapshot:
    """Observable info about enemy ships."""
    ship_id: str
    ship_name: str
    ship_type: str

    # Observable position/velocity
    position_km: Dict[str, float]
    velocity_kps: float
    velocity_vector: Dict[str, float]

    # Distance from our ships
    distance_from_closest_friendly_km: float
    closing_rate_kps: float  # Positive = closing


@dataclass
class ProjectileSnapshot:
    """Info about projectiles in flight."""
    source_ship: str
    target_ship: str
    weapon_type: str
    distance_km: float
    eta_seconds: float
    damage_gj: float


@dataclass
class AdmiralSnapshot:
    """
    Complete snapshot for Admiral decision-making.

    Contains TWO temporal views: T-15s and T=0 for trajectory analysis.
    """
    timestamp: float

    # Friendly fleet (full info)
    friendly_ships: List[FriendlyShipSnapshot]

    # Enemy fleet (observable only)
    enemy_ships: List[EnemyShipSnapshot]

    # Projectiles in flight
    projectiles: List[ProjectileSnapshot]

    # Fleet capabilities summary
    fleet_summary: str  # e.g., "2 ships: 1 Destroyer, 1 Frigate"


class LLMAdmiral:
    """
    LLM-powered fleet Admiral that issues orders to captains.
    """

    def __init__(
        self,
        config: AdmiralConfig,
        faction: str,
        client: 'CaptainClient',
        fleet_data: Dict[str, Any],
    ):
        """
        Initialize Admiral.

        Args:
            config: Admiral configuration
            faction: "alpha" or "beta"
            client: LLM client for API calls
            fleet_data: Ship specifications from fleet_ships.json
        """
        self.config = config
        self.faction = faction
        self.client = client
        self.fleet_data = fleet_data
        self.tools = get_admiral_tools()

        # State
        self.decision_count = 0
        self.order_history: List[AdmiralDecision] = []
        self.has_proposed_draw = False
        self.has_accepted_draw = False

        # Store T-15s snapshot for comparison
        self._snapshot_t_minus_15: Optional[AdmiralSnapshot] = None

        # Pending message to enemy Admiral
        self._pending_enemy_message: Optional[str] = None

        # Received messages from enemy Admiral
        self._received_enemy_messages: List[str] = []

        # All communications log (for oversight of captain messages)
        self._communications_log: List[CaptainMessage] = []

        # Ship name to ID mapping (set during setup)
        self._ship_name_to_id: Dict[str, str] = {}

    @property
    def name(self) -> str:
        """Get Admiral's name (from config or derived from model)."""
        if self.config.name:
            return self.config.name
        # Derive from model if not set
        from .fleet_config import _get_short_model_name
        return f"Admiral {_get_short_model_name(self.config.model)}"

    def set_ship_mapping(self, name_to_id: Dict[str, str]) -> None:
        """Set the mapping from ship names to IDs."""
        self._ship_name_to_id = name_to_id

    def select_personality(self, num_ships: int, verbose: bool = False) -> Dict[str, Any]:
        """
        Let the Admiral define their command personality before battle.

        Args:
            num_ships: Number of ships under command
            verbose: Whether to print selection info

        Returns:
            Dict with chosen personality info
        """
        from .tools import PERSONALITY_SELECTION_TOOLS

        # Extract a clean model name for personalization
        model_path = self.config.model.replace("openrouter/", "")
        model_name = model_path.split("/")[-1]
        model_name = "-".join(part.capitalize() for part in model_name.split("-"))

        prompt = f"""You are about to take command as Admiral of a fleet in a space combat simulation.

You command {num_ships} ships. As Admiral, you issue strategic orders to your ship captains.

Before the battle begins, define your COMMAND PERSONALITY as an Admiral.
This is about your leadership and strategic style, not individual combat tactics.

Consider:
- Your strategic philosophy (aggressive offense, calculated defense, adaptive flexibility?)
- Your leadership style (direct orders, empowering captains, micromanagement?)
- How you coordinate multiple ships (focus fire, divide and conquer, defensive formations?)
- Your decision-making approach (bold risks, conservative safety, opportunistic?)

Use the choose_personality tool to define your Admiral personality (2-4 sentences).
Be authentic to how you would command a fleet as {model_name}."""

        messages = [{"role": "user", "content": prompt}]

        # Call LLM with personality selection tool
        tool_calls = self.client.decide_with_tools(
            messages=messages,
            tools=PERSONALITY_SELECTION_TOOLS,
        )

        result = {"personality_description": None}

        for tc in tool_calls:
            if tc.name == "choose_personality":
                personality_desc = tc.arguments.get("personality_description", "")
                result["personality_description"] = personality_desc

                if personality_desc:
                    self.config.personality = personality_desc

                if verbose:
                    print(f"  [Admiral {self.name}] Defined command personality")
                    if personality_desc:
                        print(f"    {personality_desc[:80]}...")

        return result

    def capture_pre_snapshot(
        self,
        simulation: Any,
        captains: List['LLMCaptain'],
    ) -> None:
        """
        Capture T-15s snapshot for comparison.

        Called 15 seconds before checkpoint to capture pre-snapshot data.

        Args:
            simulation: Combat simulation instance
            captains: List of friendly captains
        """
        self._snapshot_t_minus_15 = self._build_snapshot(simulation, captains)

    def decide(
        self,
        simulation: Any,
        captains: List['LLMCaptain'],
        enemy_admiral: Optional['LLMAdmiral'] = None,
    ) -> AdmiralDecision:
        """
        Make fleet-level decisions using two-phase approach.

        Phase 1: Set fleet directive (overall strategy)
        Phase 2: Issue order to each ship individually

        Called at checkpoint, BEFORE captains decide.

        Args:
            simulation: Combat simulation instance
            captains: List of friendly captains
            enemy_admiral: Enemy Admiral (if exists)

        Returns:
            AdmiralDecision with orders for captains
        """
        from .prompts import build_admiral_prompt, build_admiral_ship_order_prompt

        # Build current snapshot
        snapshot_t0 = self._build_snapshot(simulation, captains)

        # Initialize decision
        decision = AdmiralDecision(timestamp=simulation.current_time)

        # Get list of active ships that need orders
        active_ships = []
        for captain in captains:
            ship_id = getattr(captain, 'ship_id', None)
            if ship_id:
                ship = simulation.get_ship(ship_id)
                if ship and not ship.is_destroyed:
                    active_ships.append({
                        'ship_id': ship_id,
                        'ship_name': captain.ship_name,
                        'captain_name': captain.name,
                        'ship_type': captain.config.ship_type,
                    })

        # PHASE 1: Get fleet directive (overall strategy)
        prompt = build_admiral_prompt(
            admiral_name=self.config.name,
            faction=self.faction,
            snapshot_t_minus_15=self._snapshot_t_minus_15,
            snapshot_t_zero=snapshot_t0,
            personality=self.config.personality,
            fleet_data=self.fleet_data,
            enemy_has_admiral=enemy_admiral is not None,
            enemy_proposed_draw=enemy_admiral.has_proposed_draw if enemy_admiral else False,
            received_messages=self._received_enemy_messages,
            communications_log=self._communications_log,
            phase="directive",  # Signal that we only want the directive
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"ADMIRAL CHECKPOINT {self.decision_count + 1}. Set your fleet directive (overall strategy). You will issue individual ship orders next."},
        ]

        # Call LLM for directive
        tool_calls = self.client.decide_with_tools(messages, self.tools)

        # Extract directive and any draw/message actions
        for call in tool_calls:
            name = call.name
            args = call.arguments

            if name == "set_fleet_directive":
                decision.fleet_directive = args.get("directive", "")
            elif name == "message_enemy_admiral":
                self._pending_enemy_message = args.get("message", "")
                decision.message_to_enemy_admiral = self._pending_enemy_message
            elif name == "propose_fleet_draw":
                self.has_proposed_draw = True
                decision.proposed_draw = True
            elif name == "accept_fleet_draw":
                self.has_accepted_draw = True
                decision.accepted_draw = True
            elif name == "reject_fleet_draw":
                decision.rejected_draw = True

        # PHASE 2: Issue order to each ship individually
        for ship_info in active_ships:
            ship_order = self._issue_ship_order(
                ship_info=ship_info,
                snapshot_t0=snapshot_t0,
                fleet_directive=decision.fleet_directive,
                simulation=simulation,
            )
            if ship_order:
                decision.fleet_orders.append(ship_order)

        # Update state
        self.decision_count += 1
        self.order_history.append(decision)

        # Clear received messages after processing
        self._received_enemy_messages.clear()

        return decision

    def _issue_ship_order(
        self,
        ship_info: dict,
        snapshot_t0: 'AdmiralSnapshot',
        fleet_directive: str,
        simulation: Any,
    ) -> Optional[AdmiralOrder]:
        """
        Issue a specific order to a single ship.

        Args:
            ship_info: Dict with ship_id, ship_name, captain_name, ship_type
            snapshot_t0: Current battle snapshot
            fleet_directive: The fleet-wide strategy
            simulation: Current simulation

        Returns:
            AdmiralOrder for this ship
        """
        from .prompts import build_admiral_ship_order_prompt

        # Build focused prompt for this ship
        prompt = build_admiral_ship_order_prompt(
            admiral_name=self.config.name,
            ship_name=ship_info['ship_name'],
            ship_type=ship_info['ship_type'],
            captain_name=ship_info['captain_name'],
            fleet_directive=fleet_directive,
            snapshot=snapshot_t0,
            personality=self.config.personality,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Issue tactical order to {ship_info['ship_name']}. You MUST call issue_order for this ship."},
        ]

        # Use only the issue_order tool for this call
        order_tool = [t for t in self.tools if t.get("function", {}).get("name") == "issue_order"]

        # Call LLM
        tool_calls = self.client.decide_with_tools(messages, order_tool)

        # Extract order
        for call in tool_calls:
            if call.name == "issue_order":
                args = call.arguments
                return AdmiralOrder(
                    target_ship_id=ship_info['ship_id'],
                    target_ship_name=ship_info['ship_name'],
                    order_text=args.get("order_text", "Continue as directed."),
                    priority=args.get("priority", "NORMAL"),
                    suggested_target=args.get("suggested_target"),
                )

        # If no order was issued, create a default order
        return AdmiralOrder(
            target_ship_id=ship_info['ship_id'],
            target_ship_name=ship_info['ship_name'],
            order_text=f"[AUTO] Continue per fleet directive: {fleet_directive[:100]}...",
            priority="NORMAL",
            suggested_target=None,
        )

    def respond_to_captain(
        self,
        captain_ship_name: str,
        question: str,
        simulation: Any,
    ) -> str:
        """
        Respond to a captain's discussion request.

        Args:
            captain_ship_name: Name of the captain's ship
            question: Captain's question
            simulation: Current simulation state

        Returns:
            Admiral's response text
        """
        from .prompts import build_admiral_response_prompt

        # Get recent decisions for context (up to last 3)
        recent_decisions = self.order_history[-3:] if self.order_history else []

        # Build context for response
        prompt = build_admiral_response_prompt(
            admiral_name=self.config.name,
            captain_ship_name=captain_ship_name,
            question=question,
            personality=self.config.personality,
            recent_decisions=recent_decisions,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Captain of {captain_ship_name} asks: {question}"},
        ]

        # Get response (no tools, just text)
        response = self.client.complete(messages)
        return response.content

    def receive_enemy_admiral_message(self, message: str) -> None:
        """Receive a message from enemy Admiral."""
        self._received_enemy_messages.append(message)

    def get_pending_enemy_message(self) -> Optional[str]:
        """Get and clear pending message to enemy Admiral."""
        msg = self._pending_enemy_message
        self._pending_enemy_message = None
        return msg

    def add_to_communications_log(self, message: CaptainMessage) -> None:
        """Add a message to the communications log (Admiral sees all)."""
        self._communications_log.append(message)

    def clear_communications_log(self) -> None:
        """Clear the communications log for new checkpoint."""
        self._communications_log.clear()

    def _build_snapshot(
        self,
        simulation: Any,
        captains: List['LLMCaptain'],
    ) -> AdmiralSnapshot:
        """Build a complete snapshot of current battle state."""
        # Get friendly ships
        friendly_snapshots = []
        for captain in captains:
            # Use ship_id to look up ship (set by battle_runner)
            ship_id = getattr(captain, 'ship_id', None)
            if not ship_id:
                continue
            ship = simulation.get_ship(ship_id)
            if ship and not ship.is_destroyed:
                friendly_snapshots.append(
                    self._build_friendly_snapshot(ship, captain, simulation)
                )

        # Get enemy ships
        enemy_snapshots = []
        if captains:
            # Get first captain's ship to find enemies
            first_ship_id = getattr(captains[0], 'ship_id', None)
            if not first_ship_id:
                first_ship_id = captains[0].config.ship_name
            for enemy in simulation.get_enemy_ships(first_ship_id):
                if not enemy.is_destroyed:
                    enemy_snapshots.append(
                        self._build_enemy_snapshot(enemy, friendly_snapshots, simulation)
                    )

        # Get projectiles
        projectile_snapshots = self._build_projectile_snapshots(simulation)

        # Build fleet summary
        ship_types = [s.ship_type for s in friendly_snapshots]
        type_counts = {}
        for t in ship_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        summary_parts = [f"{count} {stype.title()}" for stype, count in type_counts.items()]
        fleet_summary = f"{len(friendly_snapshots)} ships: {', '.join(summary_parts)}"

        return AdmiralSnapshot(
            timestamp=simulation.current_time,
            friendly_ships=friendly_snapshots,
            enemy_ships=enemy_snapshots,
            projectiles=projectile_snapshots,
            fleet_summary=fleet_summary,
        )

    def _build_friendly_snapshot(
        self,
        ship: Any,
        captain: 'LLMCaptain',
        simulation: Any,
    ) -> FriendlyShipSnapshot:
        """Build full snapshot for a friendly ship."""
        # Get ship capabilities from fleet data
        ship_spec = self.fleet_data.get("ships", {}).get(captain.config.ship_type, {})
        propulsion = ship_spec.get("propulsion", {})

        # Build weapons summary
        weapons_summary = self._build_weapons_summary(ship_spec)

        # Weapons status
        weapons_ready = []
        weapons_cooling = []
        weapons_destroyed = []
        if hasattr(ship, 'weapons'):
            for slot, weapon in ship.weapons.items():
                if slot.startswith("pd_"):
                    continue
                # Check if weapon is destroyed
                if hasattr(weapon, 'is_operational') and not weapon.is_operational:
                    weapons_destroyed.append(slot)
                elif weapon.cooldown_remaining <= 0:
                    weapons_ready.append(slot)
                else:
                    weapons_cooling.append(f"{slot} ({weapon.cooldown_remaining:.0f}s)")

        # Get targeting info
        targeted_by = []
        for enemy in simulation.get_enemy_ships(ship.ship_id):
            if hasattr(enemy, 'primary_target_id') and enemy.primary_target_id == ship.ship_id:
                targeted_by.append(enemy.ship_name if hasattr(enemy, 'ship_name') else enemy.ship_id)

        # Current maneuver
        maneuver_str = "MAINTAIN"
        if hasattr(ship, 'current_maneuver') and ship.current_maneuver:
            maneuver_str = ship.current_maneuver.maneuver_type.name

        # Position in km
        pos = ship.position
        pos_km = {"x": pos.x / 1000, "y": pos.y / 1000, "z": pos.z / 1000}

        # Velocity
        vel = ship.velocity
        vel_vector = {"x": vel.x / 1000, "y": vel.y / 1000, "z": vel.z / 1000}
        vel_kps = vel.magnitude / 1000

        return FriendlyShipSnapshot(
            ship_id=ship.ship_id,
            ship_name=ship.ship_name if hasattr(ship, 'ship_name') else ship.ship_id,
            ship_type=captain.config.ship_type,
            captain_name=captain.config.name,
            position_km=pos_km,
            velocity_kps=vel_kps,
            velocity_vector=vel_vector,
            hull_integrity=ship.hull_integrity * 100,
            delta_v_remaining=ship.remaining_delta_v_kps,
            heat_percent=ship.heat_percent if hasattr(ship, 'heat_percent') else 0,
            max_acceleration_g=propulsion.get("combat_acceleration_g", 2.0),
            max_delta_v=propulsion.get("delta_v_kps", 500),
            weapons_summary=weapons_summary,
            weapons_ready=weapons_ready,
            weapons_cooling=weapons_cooling,
            weapons_destroyed=weapons_destroyed,
            current_maneuver=maneuver_str,
            current_target=ship.primary_target_id if hasattr(ship, 'primary_target_id') else None,
            radiators_extended=ship.radiators_extended if hasattr(ship, 'radiators_extended') else False,
            targeted_by=targeted_by,
        )

    def _build_enemy_snapshot(
        self,
        ship: Any,
        friendly_ships: List[FriendlyShipSnapshot],
        simulation: Any,
    ) -> EnemyShipSnapshot:
        """Build observable snapshot for an enemy ship."""
        pos = ship.position
        pos_km = {"x": pos.x / 1000, "y": pos.y / 1000, "z": pos.z / 1000}

        vel = ship.velocity
        vel_vector = {"x": vel.x / 1000, "y": vel.y / 1000, "z": vel.z / 1000}
        vel_kps = vel.magnitude / 1000

        # Find closest friendly and closing rate
        min_dist = float('inf')
        closing_rate = 0.0
        for friendly in friendly_ships:
            # Calculate distance
            dx = pos_km["x"] - friendly.position_km["x"]
            dy = pos_km["y"] - friendly.position_km["y"]
            dz = pos_km["z"] - friendly.position_km["z"]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5

            if dist < min_dist:
                min_dist = dist
                # Calculate closing rate (positive = closing)
                dvx = vel_vector["x"] - friendly.velocity_vector["x"]
                dvy = vel_vector["y"] - friendly.velocity_vector["y"]
                dvz = vel_vector["z"] - friendly.velocity_vector["z"]
                # Dot product of relative velocity with direction vector
                if dist > 0:
                    closing_rate = -(dx*dvx + dy*dvy + dz*dvz) / dist

        return EnemyShipSnapshot(
            ship_id=ship.ship_id,
            ship_name=ship.ship_name if hasattr(ship, 'ship_name') else ship.ship_id,
            ship_type=ship.ship_type if hasattr(ship, 'ship_type') else "unknown",
            position_km=pos_km,
            velocity_kps=vel_kps,
            velocity_vector=vel_vector,
            distance_from_closest_friendly_km=min_dist,
            closing_rate_kps=closing_rate,
        )

    def _build_projectile_snapshots(self, simulation: Any) -> List[ProjectileSnapshot]:
        """Build snapshots of projectiles in flight."""
        snapshots = []

        for proj in simulation.projectiles:
            # Calculate ETA based on distance and velocity
            if hasattr(proj, 'target_ship_id') and proj.target_ship_id:
                target = simulation.get_ship(proj.target_ship_id)
                if target:
                    dist = proj.projectile.distance_to(target.position) / 1000  # km
                    rel_vel = proj.projectile.velocity.magnitude / 1000  # km/s
                    eta = dist / rel_vel if rel_vel > 0 else 999

                    snapshots.append(ProjectileSnapshot(
                        source_ship=proj.source_ship_id,
                        target_ship=proj.target_ship_id,
                        weapon_type="Coilgun",  # Could be more specific
                        distance_km=dist,
                        eta_seconds=eta,
                        damage_gj=proj.projectile.kinetic_energy_gj,
                    ))

        return snapshots

    def _build_weapons_summary(self, ship_spec: Dict[str, Any]) -> str:
        """Build a human-readable weapons summary from ship spec."""
        weapons = ship_spec.get("weapons", [])
        weapon_types = self.fleet_data.get("weapon_types", {})

        counts = {}
        for weapon in weapons:
            wtype = weapon.get("type", "")
            if wtype.startswith("pd_"):
                continue  # Skip point defense

            # Get weapon stats
            wspec = weapon_types.get(wtype, {})
            damage = wspec.get("kinetic_energy_gj", 0)

            # Clean up name
            if "spinal" in wtype:
                name = f"Spinal ({damage:.1f} GJ)"
            elif "heavy" in wtype:
                name = f"Heavy Coilgun ({damage:.1f} GJ)"
            elif "coilgun" in wtype:
                name = f"Coilgun ({damage:.1f} GJ)"
            else:
                name = wtype

            counts[name] = counts.get(name, 0) + 1

        parts = [f"{count}x {name}" for name, count in counts.items()]
        return ", ".join(parts) if parts else "No weapons"

    def _execute_tools(
        self,
        tool_calls: List[Any],
        timestamp: float,
    ) -> AdmiralDecision:
        """Execute tool calls and build decision."""
        decision = AdmiralDecision(timestamp=timestamp)

        for call in tool_calls:
            name = call.name
            args = call.arguments

            if name == "issue_order":
                ship_name = args.get("ship_name", "")
                ship_id = self._ship_name_to_id.get(ship_name, ship_name)

                order = AdmiralOrder(
                    target_ship_id=ship_id,
                    target_ship_name=ship_name,
                    order_text=args.get("order_text", ""),
                    priority=args.get("priority", "NORMAL"),
                    suggested_target=args.get("suggested_target"),
                )
                decision.fleet_orders.append(order)

            elif name == "set_fleet_directive":
                decision.fleet_directive = args.get("directive", "")

            elif name == "message_enemy_admiral":
                self._pending_enemy_message = args.get("message", "")
                decision.message_to_enemy_admiral = self._pending_enemy_message

            elif name == "propose_fleet_draw":
                self.has_proposed_draw = True
                decision.proposed_draw = True

            elif name == "accept_fleet_draw":
                self.has_accepted_draw = True
                decision.accepted_draw = True

            elif name == "reject_fleet_draw":
                decision.rejected_draw = True

        return decision
