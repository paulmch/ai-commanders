#!/usr/bin/env python3
"""
Combat Script System for AI Commanders Space Battle Simulator.

This module defines pre-programmed sequences of actions that ships take,
simulating what an LLM captain might order over time. Used primarily for
testing the simulation without LLM integration.

Key classes:
- CombatScript: A sequence of (time, action) tuples with conditional logic
- ScriptExecutor: Executes scripts step-by-step against a ShipCombatState
- ScriptGenerator: Creates varied combat scripts for testing

Pre-built scripts for testing:
- AggressiveApproach: Close with enemy, firing at opportunity
- DefensiveStance: Maintain distance, maximize armor, return fire
- HitAndRun: Fast approach, fire everything at closest range, escape
- TorpedoRun: Approach to torpedo range, launch salvo, retreat
- EvasivePattern: Continuous evasive maneuvers while engaging

Each script includes tactical reasoning as comments for learning purposes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List, Tuple, Any, Dict, Union

# Try/except import pattern for relative/absolute imports
try:
    from .physics import Vector3D
    from .maneuvers import (
        Maneuver, ManeuverStatus, ManeuverResult,
        BurnToward, BurnAway, MatchVelocity, BrakingBurn, AccelerationBurn,
        RotateToFace, RotateToBroadside, RotateToRetreat, RotateToAngle,
        FlipAndBurn, EvasiveJink, SpiralApproach, BreakTurn,
        ManeuverExecutor, ManeuverPlanner
    )
except ImportError:
    from physics import Vector3D
    from maneuvers import (
        Maneuver, ManeuverStatus, ManeuverResult,
        BurnToward, BurnAway, MatchVelocity, BrakingBurn, AccelerationBurn,
        RotateToFace, RotateToBroadside, RotateToRetreat, RotateToAngle,
        FlipAndBurn, EvasiveJink, SpiralApproach, BreakTurn,
        ManeuverExecutor, ManeuverPlanner
    )

# Import ShipCombatState for type hints (optional at runtime)
try:
    from .simulation import ShipCombatState
except ImportError:
    try:
        from simulation import ShipCombatState
    except ImportError:
        # Define minimal stub if simulation module unavailable
        ShipCombatState = Any  # type: ignore


# =============================================================================
# ACTION TYPES
# =============================================================================

class ActionType(Enum):
    """Types of actions that can be scheduled in a combat script."""
    MANEUVER = auto()           # Execute a maneuver from maneuvers.py
    FIRE_COILGUN = auto()       # Fire coilgun at target
    LAUNCH_TORPEDO = auto()     # Launch torpedo at target
    EXTEND_RADIATORS = auto()   # Extend radiators for cooling
    RETRACT_RADIATORS = auto()  # Retract radiators for protection
    SET_TARGET = auto()         # Set primary engagement target
    HOLD_FIRE = auto()          # Stop firing weapons
    EVALUATE = auto()           # Trigger condition evaluation


class ConditionType(Enum):
    """Types of conditions for conditional actions."""
    HEALTH_BELOW = auto()       # Hull integrity below threshold
    HEALTH_ABOVE = auto()       # Hull integrity above threshold
    HEAT_ABOVE = auto()         # Heat level above threshold
    HEAT_BELOW = auto()         # Heat level below threshold
    RANGE_BELOW = auto()        # Distance to target below threshold (km)
    RANGE_ABOVE = auto()        # Distance to target above threshold (km)
    DELTA_V_BELOW = auto()      # Remaining delta-v below threshold (km/s)
    AMMO_BELOW = auto()         # Ammunition below threshold (count)
    ENEMY_COUNT = auto()        # Number of enemies equals value
    INCOMING_TORPEDO = auto()   # Torpedo inbound


# =============================================================================
# SCRIPT ACTION
# =============================================================================

@dataclass
class Condition:
    """
    A condition that must be met for an action to execute.

    Attributes:
        condition_type: Type of condition to check
        threshold: Numeric threshold for comparison
        comparison: 'lt', 'gt', 'eq', 'le', 'ge' for comparison type
    """
    condition_type: ConditionType
    threshold: float = 0.0
    comparison: str = "lt"  # less than by default

    def evaluate(self, ship: 'ShipCombatState', context: Dict[str, Any]) -> bool:
        """
        Evaluate this condition against current ship state.

        Args:
            ship: The ship to evaluate against
            context: Additional context (target info, etc.)

        Returns:
            True if condition is met, False otherwise
        """
        value = self._get_value(ship, context)

        if self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "eq":
            return abs(value - self.threshold) < 0.001
        elif self.comparison == "le":
            return value <= self.threshold
        elif self.comparison == "ge":
            return value >= self.threshold
        return False

    def _get_value(self, ship: 'ShipCombatState', context: Dict[str, Any]) -> float:
        """Get the value to compare against threshold."""
        if self.condition_type == ConditionType.HEALTH_BELOW:
            return ship.hull_integrity if hasattr(ship, 'hull_integrity') else 100.0
        elif self.condition_type == ConditionType.HEALTH_ABOVE:
            return ship.hull_integrity if hasattr(ship, 'hull_integrity') else 100.0
        elif self.condition_type == ConditionType.HEAT_ABOVE:
            return ship.heat_percent if hasattr(ship, 'heat_percent') else 0.0
        elif self.condition_type == ConditionType.HEAT_BELOW:
            return ship.heat_percent if hasattr(ship, 'heat_percent') else 0.0
        elif self.condition_type == ConditionType.RANGE_BELOW:
            return context.get('target_range_km', float('inf'))
        elif self.condition_type == ConditionType.RANGE_ABOVE:
            return context.get('target_range_km', float('inf'))
        elif self.condition_type == ConditionType.DELTA_V_BELOW:
            return ship.remaining_delta_v_kps if hasattr(ship, 'remaining_delta_v_kps') else 10.0
        elif self.condition_type == ConditionType.AMMO_BELOW:
            total_ammo = sum(
                ws.ammo_remaining for ws in ship.weapons.values()
            ) if hasattr(ship, 'weapons') else 100
            return float(total_ammo)
        elif self.condition_type == ConditionType.ENEMY_COUNT:
            return float(context.get('enemy_count', 0))
        elif self.condition_type == ConditionType.INCOMING_TORPEDO:
            return float(context.get('incoming_torpedoes', 0))
        return 0.0


@dataclass
class ScriptAction:
    """
    A single action in a combat script.

    Actions can be maneuvers, weapon commands, or system commands.
    Conditional actions only execute if their condition is met.

    Attributes:
        time_offset: Time offset from script start (seconds)
        action_type: Type of action to perform
        parameters: Action-specific parameters
        condition: Optional condition that must be met
        description: Human-readable description of this action
        tactical_note: Explanation of tactical reasoning
    """
    time_offset: float
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Condition] = None
    description: str = ""
    tactical_note: str = ""

    def should_execute(self, ship: 'ShipCombatState', context: Dict[str, Any]) -> bool:
        """Check if this action should execute given current state."""
        if self.condition is None:
            return True
        return self.condition.evaluate(ship, context)

    def create_maneuver(self, ship: 'ShipCombatState', context: Dict[str, Any]) -> Optional[Maneuver]:
        """
        Create a maneuver object if this action is a maneuver.

        Args:
            ship: Current ship state
            context: Context including target info

        Returns:
            Maneuver object or None if not a maneuver action
        """
        if self.action_type != ActionType.MANEUVER:
            return None

        maneuver_class = self.parameters.get('maneuver_class')
        if maneuver_class is None:
            return None

        # Get target position from context if needed
        target_pos = context.get('target_position', Vector3D.zero())
        target_vel = context.get('target_velocity', Vector3D.zero())

        # Build maneuver parameters
        params = dict(self.parameters)
        params.pop('maneuver_class', None)

        # Replace placeholder values
        if 'target_position' in params and params['target_position'] == 'TARGET':
            params['target_position'] = target_pos
        if 'target_velocity' in params and params['target_velocity'] == 'TARGET':
            params['target_velocity'] = target_vel

        try:
            return maneuver_class(**params)
        except Exception as e:
            print(f"Error creating maneuver: {e}")
            return None


# =============================================================================
# COMBAT SCRIPT
# =============================================================================

@dataclass
class CombatScript:
    """
    A pre-programmed sequence of combat actions.

    Scripts define time-based action sequences that simulate tactical
    decision-making by an LLM captain. Used for testing and demonstration.

    Attributes:
        name: Script identifier
        description: Human-readable description
        actions: List of (time, action) tuples sorted by time
        total_duration: Total duration of the script (seconds)
        repeat: Whether to loop the script
        base_aggressiveness: Base aggression level (0.0 to 1.0)
    """
    name: str
    description: str = ""
    actions: List[ScriptAction] = field(default_factory=list)
    total_duration: float = 120.0
    repeat: bool = False
    base_aggressiveness: float = 0.5

    def __post_init__(self):
        """Sort actions by time offset after initialization."""
        self.actions = sorted(self.actions, key=lambda a: a.time_offset)

    def add_action(
        self,
        time_offset: float,
        action_type: ActionType,
        parameters: Optional[Dict[str, Any]] = None,
        condition: Optional[Condition] = None,
        description: str = "",
        tactical_note: str = ""
    ) -> None:
        """
        Add an action to the script.

        Args:
            time_offset: When to execute (seconds from start)
            action_type: Type of action
            parameters: Action parameters
            condition: Optional execution condition
            description: Human-readable description
            tactical_note: Tactical reasoning explanation
        """
        action = ScriptAction(
            time_offset=time_offset,
            action_type=action_type,
            parameters=parameters or {},
            condition=condition,
            description=description,
            tactical_note=tactical_note
        )
        self.actions.append(action)
        self.actions = sorted(self.actions, key=lambda a: a.time_offset)

    def get_actions_at_time(self, time: float, tolerance: float = 0.5) -> List[ScriptAction]:
        """
        Get all actions scheduled near a given time.

        Args:
            time: Current time offset from script start
            tolerance: Time window for matching (seconds)

        Returns:
            List of actions within the time window
        """
        return [
            a for a in self.actions
            if abs(a.time_offset - time) <= tolerance
        ]

    def get_next_action_time(self, current_time: float) -> Optional[float]:
        """Get the time of the next scheduled action."""
        future_actions = [a for a in self.actions if a.time_offset > current_time]
        if future_actions:
            return future_actions[0].time_offset
        elif self.repeat:
            return self.actions[0].time_offset if self.actions else None
        return None


# =============================================================================
# SCRIPT EXECUTOR
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of executing a script action."""
    action: ScriptAction
    executed: bool
    maneuver: Optional[Maneuver] = None
    command: Optional[Dict[str, Any]] = None
    message: str = ""


class ScriptExecutor:
    """
    Executes combat scripts against a ship state.

    Takes a CombatScript and ShipCombatState, executes actions at
    specified times, handles conditional logic, and reports actions taken.

    Attributes:
        script: The combat script being executed
        start_time: Simulation time when script started
        last_execution_time: Time of last action execution
        is_active: Whether the script is currently running
    """

    def __init__(self, script: CombatScript) -> None:
        """
        Initialize the script executor.

        Args:
            script: The combat script to execute
        """
        self.script = script
        self.start_time: float = 0.0
        self.last_execution_time: float = 0.0
        self.is_active: bool = False
        self._executed_actions: set = set()
        self._current_maneuver: Optional[Maneuver] = None
        self._maneuver_executor = ManeuverExecutor()

    def start(self, current_time: float) -> None:
        """Start executing the script from the current time."""
        self.start_time = current_time
        self.last_execution_time = current_time
        self.is_active = True
        self._executed_actions = set()

    def stop(self) -> None:
        """Stop script execution."""
        self.is_active = False
        self._current_maneuver = None

    def reset(self) -> None:
        """Reset the executor to initial state."""
        self.start_time = 0.0
        self.last_execution_time = 0.0
        self.is_active = False
        self._executed_actions = set()
        self._current_maneuver = None

    def update(
        self,
        ship: 'ShipCombatState',
        current_time: float,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExecutionResult]:
        """
        Update the script executor and execute due actions.

        Args:
            ship: Current ship combat state
            current_time: Current simulation time
            context: Additional context (targets, enemies, etc.)

        Returns:
            List of execution results for this update
        """
        if not self.is_active:
            return []

        context = context or {}
        results: List[ExecutionResult] = []

        # Calculate script-relative time
        script_time = current_time - self.start_time

        # Handle script looping
        if script_time > self.script.total_duration:
            if self.script.repeat:
                # Reset for next loop
                self.start_time = current_time
                self._executed_actions = set()
                script_time = 0.0
            else:
                self.is_active = False
                return results

        # Get actions that should execute now
        for action in self.script.actions:
            action_id = id(action)

            # Skip already executed (non-repeating) actions
            if action_id in self._executed_actions:
                continue

            # Check if action is due
            if action.time_offset <= script_time:
                # Check condition
                if not action.should_execute(ship, context):
                    # Mark as executed even if condition failed
                    self._executed_actions.add(action_id)
                    results.append(ExecutionResult(
                        action=action,
                        executed=False,
                        message=f"Condition not met: {action.description}"
                    ))
                    continue

                # Execute the action
                result = self._execute_action(action, ship, context)
                results.append(result)
                self._executed_actions.add(action_id)

        self.last_execution_time = current_time
        return results

    def _execute_action(
        self,
        action: ScriptAction,
        ship: 'ShipCombatState',
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute a single action and return the result."""

        if action.action_type == ActionType.MANEUVER:
            maneuver = action.create_maneuver(ship, context)
            if maneuver:
                self._current_maneuver = maneuver
                self._maneuver_executor.set_maneuver(maneuver)
                return ExecutionResult(
                    action=action,
                    executed=True,
                    maneuver=maneuver,
                    message=f"Started maneuver: {maneuver.name}"
                )
            return ExecutionResult(
                action=action,
                executed=False,
                message="Failed to create maneuver"
            )

        elif action.action_type == ActionType.FIRE_COILGUN:
            target_id = action.parameters.get('target_id', context.get('primary_target_id'))
            weapon_slot = action.parameters.get('weapon_slot', 'spinal')
            command = {
                'type': 'fire_at',
                'weapon_slot': weapon_slot,
                'target_id': target_id
            }
            return ExecutionResult(
                action=action,
                executed=True,
                command=command,
                message=f"Fire command: {weapon_slot} at {target_id}"
            )

        elif action.action_type == ActionType.LAUNCH_TORPEDO:
            target_id = action.parameters.get('target_id', context.get('primary_target_id'))
            command = {
                'type': 'launch_torpedo',
                'target_id': target_id
            }
            return ExecutionResult(
                action=action,
                executed=True,
                command=command,
                message=f"Torpedo launch at {target_id}"
            )

        elif action.action_type == ActionType.EXTEND_RADIATORS:
            command = {'type': 'set_radiators', 'extend': True}
            return ExecutionResult(
                action=action,
                executed=True,
                command=command,
                message="Extending radiators"
            )

        elif action.action_type == ActionType.RETRACT_RADIATORS:
            command = {'type': 'set_radiators', 'extend': False}
            return ExecutionResult(
                action=action,
                executed=True,
                command=command,
                message="Retracting radiators"
            )

        elif action.action_type == ActionType.SET_TARGET:
            target_id = action.parameters.get('target_id')
            command = {'type': 'set_target', 'target_id': target_id}
            return ExecutionResult(
                action=action,
                executed=True,
                command=command,
                message=f"Target set: {target_id}"
            )

        elif action.action_type == ActionType.HOLD_FIRE:
            command = {'type': 'hold_fire'}
            return ExecutionResult(
                action=action,
                executed=True,
                command=command,
                message="Holding fire"
            )

        elif action.action_type == ActionType.EVALUATE:
            # Evaluation action - just triggers condition checks
            return ExecutionResult(
                action=action,
                executed=True,
                message="Evaluation point"
            )

        return ExecutionResult(
            action=action,
            executed=False,
            message=f"Unknown action type: {action.action_type}"
        )

    def get_current_maneuver_result(
        self,
        ship: 'ShipCombatState',
        dt: float
    ) -> Optional[ManeuverResult]:
        """
        Get the current maneuver execution result.

        Args:
            ship: Current ship state
            dt: Time step

        Returns:
            ManeuverResult if a maneuver is active, None otherwise
        """
        if self._maneuver_executor.is_idle:
            return None

        # Get kinematic state from ship
        kinematic_state = getattr(ship, 'kinematic_state', None)
        if kinematic_state is None:
            return None

        return self._maneuver_executor.update(kinematic_state, dt)

    @property
    def progress(self) -> float:
        """Get script execution progress as percentage."""
        if not self.is_active:
            return 0.0
        script_time = self.last_execution_time - self.start_time
        return min(100.0, (script_time / self.script.total_duration) * 100)


# =============================================================================
# PRE-BUILT COMBAT SCRIPTS
# =============================================================================

def create_aggressive_approach_script() -> CombatScript:
    """
    Create an aggressive approach combat script.

    Tactical Intent: Close with enemy rapidly, engage at optimal range.

    Sequence:
    - T+0s: Rotate to face enemy
    - T+20s: Full burn toward enemy
    - T+40s: Fire coilgun salvo
    - T+60s: Continue burn, fire again
    - T+80s: Launch torpedo at close range
    - T+100s: Continue engagement until delta-v depleted
    """
    script = CombatScript(
        name="AggressiveApproach",
        description="Close with enemy rapidly while firing at opportunity",
        total_duration=120.0,
        repeat=False,
        base_aggressiveness=0.9
    )

    # T+0s: Initial orientation toward enemy
    # Tactical: Must face target for spinal weapons
    script.add_action(
        time_offset=0.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': RotateToFace,
            'target_position': 'TARGET'
        },
        description="Rotate to face enemy",
        tactical_note="Spinal weapons require nose-on orientation. Priority is getting guns on target."
    )

    # T+20s: Full burn toward target
    # Tactical: Close range quickly to maximize hit probability and minimize enemy reaction time
    script.add_action(
        time_offset=20.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': BurnToward,
            'target_position': 'TARGET',
            'throttle': 1.0,
            'max_duration': 60.0
        },
        description="Full burn toward enemy",
        tactical_note="Closing rapidly reduces time for enemy evasion and improves our hit probability."
    )

    # T+40s: First coilgun salvo
    # Tactical: Fire at opportunity even at range - every hit counts
    script.add_action(
        time_offset=40.0,
        action_type=ActionType.FIRE_COILGUN,
        parameters={'weapon_slot': 'spinal'},
        description="Fire coilgun salvo",
        tactical_note="Opening salvo at range. Hit probability lower but forces enemy to react."
    )

    # T+60s: Continue engagement
    # Tactical: Sustained fire while closing
    script.add_action(
        time_offset=60.0,
        action_type=ActionType.FIRE_COILGUN,
        parameters={'weapon_slot': 'spinal'},
        description="Continue coilgun fire",
        tactical_note="Maintain fire pressure. Closer range means higher hit probability."
    )

    # T+80s: Launch torpedo at closer range
    # Tactical: Torpedoes most effective at medium range where they can adjust
    script.add_action(
        time_offset=80.0,
        action_type=ActionType.LAUNCH_TORPEDO,
        description="Launch torpedo",
        tactical_note="Torpedo launch at medium range gives it fuel for terminal maneuvering.",
        condition=Condition(
            condition_type=ConditionType.RANGE_BELOW,
            threshold=500.0,  # km
            comparison="lt"
        )
    )

    # T+100s: Final approach fire
    script.add_action(
        time_offset=100.0,
        action_type=ActionType.FIRE_COILGUN,
        parameters={'weapon_slot': 'spinal'},
        description="Close range fire",
        tactical_note="Point-blank engagement - maximum damage potential."
    )

    # Conditional: If health drops below 50%, switch to evasive
    script.add_action(
        time_offset=50.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': EvasiveJink,
            'duration': 20.0,
            'jink_interval': 2.0,
            'throttle': 1.0
        },
        description="Emergency evasive (if damaged)",
        tactical_note="Taking heavy damage - break off attack run to survive.",
        condition=Condition(
            condition_type=ConditionType.HEALTH_BELOW,
            threshold=50.0,
            comparison="lt"
        )
    )

    return script


def create_defensive_stance_script() -> CombatScript:
    """
    Create a defensive stance combat script.

    Tactical Intent: Maintain optimal range, maximize armor protection, return fire.

    Sequence:
    - T+0s: Extend radiators (prepare for sustained combat)
    - T+20s: Rotate to broadside (maximize armor profile)
    - T+40s: Fire at opportunity
    - T+60s: Evasive jink if incoming fire detected
    - T+80s: Retract radiators if threatened
    """
    script = CombatScript(
        name="DefensiveStance",
        description="Defensive posture prioritizing survival and armor protection",
        total_duration=120.0,
        repeat=True,  # Continuous defensive behavior
        base_aggressiveness=0.3
    )

    # T+0s: Extend radiators
    # Tactical: Heat management is critical for sustained defensive operations
    script.add_action(
        time_offset=0.0,
        action_type=ActionType.EXTEND_RADIATORS,
        description="Extend radiators",
        tactical_note="Heat management priority in defensive stance - we may need sustained burns."
    )

    # T+20s: Rotate to broadside
    # Tactical: Present maximum armored surface, not vulnerable nose
    script.add_action(
        time_offset=20.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': RotateToBroadside,
            'target_position': 'TARGET',
            'prefer_port': True
        },
        description="Rotate to broadside",
        tactical_note="Broadside presents our thickest armor section to incoming fire."
    )

    # T+40s: Return fire
    # Tactical: Defensive doesn't mean passive - punish enemy aggression
    script.add_action(
        time_offset=40.0,
        action_type=ActionType.FIRE_COILGUN,
        parameters={'weapon_slot': 'spinal'},
        description="Return fire",
        tactical_note="Maintain offensive pressure even while defending.",
        # Only fire if we have a good shot
        condition=Condition(
            condition_type=ConditionType.RANGE_BELOW,
            threshold=300.0,  # km - defensive engagement range
            comparison="lt"
        )
    )

    # T+60s: Evasive if torpedoes inbound
    # Tactical: Torpedoes are existential threats - evasion mandatory
    script.add_action(
        time_offset=60.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': EvasiveJink,
            'duration': 15.0,
            'jink_interval': 2.0,
            'throttle': 0.8
        },
        description="Evasive maneuvers (torpedo defense)",
        tactical_note="Torpedo inbound - random evasion degrades their targeting solution.",
        condition=Condition(
            condition_type=ConditionType.INCOMING_TORPEDO,
            threshold=0.5,  # Any torpedoes
            comparison="gt"
        )
    )

    # T+80s: Retract radiators if under fire
    # Tactical: Radiators are vulnerable - protect them when incoming
    script.add_action(
        time_offset=80.0,
        action_type=ActionType.RETRACT_RADIATORS,
        description="Retract radiators (protect from fire)",
        tactical_note="Enemy engaging - retract radiators to prevent easy damage.",
        condition=Condition(
            condition_type=ConditionType.INCOMING_TORPEDO,
            threshold=0.5,
            comparison="gt"
        )
    )

    # T+100s: Maintain distance burn
    # Tactical: Keep enemy at optimal engagement range
    script.add_action(
        time_offset=100.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': BurnAway,
            'target_position': 'TARGET',
            'throttle': 0.5,
            'max_duration': 20.0
        },
        description="Maintain distance",
        tactical_note="Enemy closing - burn away to maintain defensive range.",
        condition=Condition(
            condition_type=ConditionType.RANGE_BELOW,
            threshold=100.0,  # km
            comparison="lt"
        )
    )

    return script


def create_hit_and_run_script() -> CombatScript:
    """
    Create a hit-and-run combat script.

    Tactical Intent: Fast approach, deliver maximum damage at close range, escape.

    Sequence:
    - T+0s: Full burn toward target
    - T+40s: Fire everything at closest approach
    - T+60s: Flip and burn away
    - T+100s: Check for pursuit, continue escape
    """
    script = CombatScript(
        name="HitAndRun",
        description="Fast attack run with rapid disengagement",
        total_duration=120.0,
        repeat=False,
        base_aggressiveness=0.7
    )

    # T+0s: Aggressive approach
    # Tactical: Close rapidly using spiral to complicate enemy targeting
    script.add_action(
        time_offset=0.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': SpiralApproach,
            'target_position': 'TARGET',
            'approach_throttle': 1.0,
            'spiral_rate_deg_per_s': 20.0,
            'max_duration': 60.0
        },
        description="Spiral approach at full speed",
        tactical_note="Spiral pattern makes us harder to hit while we close."
    )

    # T+30s: Fire coilgun during approach
    script.add_action(
        time_offset=30.0,
        action_type=ActionType.FIRE_COILGUN,
        parameters={'weapon_slot': 'spinal'},
        description="Fire during approach",
        tactical_note="Harassing fire - forces enemy to evade, disrupts their plan."
    )

    # T+40s: Maximum firepower at closest approach
    # Tactical: This is our kill shot - everything we have
    script.add_action(
        time_offset=40.0,
        action_type=ActionType.FIRE_COILGUN,
        parameters={'weapon_slot': 'spinal'},
        description="Fire at close range",
        tactical_note="CLOSEST APPROACH - maximum probability of hits."
    )

    # T+45s: Launch torpedo at close range
    script.add_action(
        time_offset=45.0,
        action_type=ActionType.LAUNCH_TORPEDO,
        description="Launch torpedo at close range",
        tactical_note="Torpedo at minimum range - almost guaranteed hit."
    )

    # T+50s: Second torpedo if available
    script.add_action(
        time_offset=50.0,
        action_type=ActionType.LAUNCH_TORPEDO,
        description="Second torpedo salvo",
        tactical_note="Overwhelm point defenses with multiple torpedoes."
    )

    # T+60s: Flip and escape
    # Tactical: We've done our damage - time to leave
    script.add_action(
        time_offset=60.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': FlipAndBurn,
            'target_speed_ms': 0.0,  # We want to reverse direction
            'throttle': 1.0
        },
        description="Flip and burn - begin escape",
        tactical_note="Attack complete - reverse thrust to open distance."
    )

    # T+80s: Continue escape burn
    script.add_action(
        time_offset=80.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': BurnAway,
            'target_position': 'TARGET',
            'throttle': 1.0,
            'max_duration': 40.0
        },
        description="Full escape burn",
        tactical_note="Maximum thrust away - open range before enemy can respond."
    )

    # T+100s: Evasive during escape if pursued
    script.add_action(
        time_offset=100.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': EvasiveJink,
            'duration': 20.0,
            'jink_interval': 3.0,
            'throttle': 0.7
        },
        description="Evasive escape pattern",
        tactical_note="Add randomness to escape vector - complicates pursuit.",
        condition=Condition(
            condition_type=ConditionType.RANGE_BELOW,
            threshold=200.0,  # Still being chased
            comparison="lt"
        )
    )

    return script


def create_torpedo_run_script() -> CombatScript:
    """
    Create a torpedo attack run script.

    Tactical Intent: Approach to optimal torpedo range, launch salvo, retreat.

    Sequence:
    - T+0s: Approach to torpedo range
    - T+30s: Launch all torpedoes
    - T+60s: Turn and run
    - T+90s: Monitor torpedo status
    """
    script = CombatScript(
        name="TorpedoRun",
        description="Dedicated torpedo attack with coordinated salvo",
        total_duration=120.0,
        repeat=False,
        base_aggressiveness=0.6
    )

    # T+0s: Close to torpedo range
    # Tactical: Torpedoes most effective at specific range - not too close, not too far
    script.add_action(
        time_offset=0.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': BurnToward,
            'target_position': 'TARGET',
            'throttle': 0.8,
            'max_duration': 45.0
        },
        description="Approach to torpedo range",
        tactical_note="Close to ~500km - optimal torpedo launch distance."
    )

    # T+30s: First torpedo
    # Tactical: Stagger launches slightly to complicate PD response
    script.add_action(
        time_offset=30.0,
        action_type=ActionType.LAUNCH_TORPEDO,
        description="Launch torpedo 1",
        tactical_note="First torpedo of salvo - targeting solutions calculated."
    )

    # T+35s: Second torpedo
    script.add_action(
        time_offset=35.0,
        action_type=ActionType.LAUNCH_TORPEDO,
        description="Launch torpedo 2",
        tactical_note="Staggered launch overwhelms point defense."
    )

    # T+40s: Third torpedo
    script.add_action(
        time_offset=40.0,
        action_type=ActionType.LAUNCH_TORPEDO,
        description="Launch torpedo 3",
        tactical_note="Three torpedoes in flight - high probability of at least one hit."
    )

    # T+45s: Fourth torpedo
    script.add_action(
        time_offset=45.0,
        action_type=ActionType.LAUNCH_TORPEDO,
        description="Launch torpedo 4",
        tactical_note="Full salvo deployed - maximum saturation."
    )

    # T+60s: Begin retreat
    # Tactical: Torpedoes away - get out before enemy retaliates
    script.add_action(
        time_offset=60.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': RotateToRetreat,
            'target_position': 'TARGET'
        },
        description="Rotate for retreat",
        tactical_note="Torpedoes launched - orient for escape."
    )

    # T+70s: Escape burn
    script.add_action(
        time_offset=70.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': BurnAway,
            'target_position': 'TARGET',
            'throttle': 1.0,
            'max_duration': 50.0
        },
        description="Escape burn",
        tactical_note="Full thrust away - torpedoes will do the work."
    )

    # T+90s: Monitor and evade
    # Tactical: Watch for retaliatory torpedo launch
    script.add_action(
        time_offset=90.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': EvasiveJink,
            'duration': 15.0,
            'jink_interval': 3.0,
            'throttle': 0.6
        },
        description="Evasive during retreat",
        tactical_note="Unpredictable escape pattern - avoid retaliation.",
        condition=Condition(
            condition_type=ConditionType.INCOMING_TORPEDO,
            threshold=0.5,
            comparison="gt"
        )
    )

    return script


def create_evasive_pattern_script() -> CombatScript:
    """
    Create an evasive pattern combat script.

    Tactical Intent: Maximize survivability through continuous evasion.

    Sequence:
    - T+0s: Begin evasive jinks
    - T+20s: Random direction change
    - T+40s: Spiral if torpedo incoming
    - T+60s: Return fire when safe
    """
    script = CombatScript(
        name="EvasivePattern",
        description="Maximum evasion while maintaining some offensive capability",
        total_duration=100.0,
        repeat=True,  # Continuous evasion
        base_aggressiveness=0.2
    )

    # T+0s: Begin evasive maneuvers
    # Tactical: Immediately become hard to hit
    script.add_action(
        time_offset=0.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': EvasiveJink,
            'duration': 25.0,
            'jink_interval': 2.5,
            'throttle': 0.9
        },
        description="Begin evasive jinks",
        tactical_note="Random thrust changes make fire control solutions invalid."
    )

    # T+20s: Break turn to change engagement angle
    # Tactical: Major direction change invalidates enemy lead calculations
    script.add_action(
        time_offset=20.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': BreakTurn,
            'turn_angle_deg': 60.0,
            'turn_direction': 'right',
            'burn_duration': 8.0,
            'throttle': 1.0
        },
        description="Break turn - change engagement plane",
        tactical_note="Major vector change - enemy must recalculate approach."
    )

    # T+40s: Hard evasion if torpedo incoming
    # Tactical: Torpedoes track - need continuous maximum evasion
    script.add_action(
        time_offset=40.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': SpiralApproach,  # Using spiral as spiral evasion
            'target_position': 'TARGET',
            'approach_throttle': 0.8,
            'spiral_rate_deg_per_s': 45.0,  # Fast spiral
            'spiral_amplitude': 0.5,
            'max_duration': 20.0
        },
        description="Spiral evasion (torpedo defense)",
        tactical_note="Spiral degrades torpedo tracking - their guidance can't match our random motion.",
        condition=Condition(
            condition_type=ConditionType.INCOMING_TORPEDO,
            threshold=0.5,
            comparison="gt"
        )
    )

    # T+60s: Return fire during evasion
    # Tactical: Don't be entirely passive - punish enemy when possible
    script.add_action(
        time_offset=60.0,
        action_type=ActionType.FIRE_COILGUN,
        parameters={'weapon_slot': 'spinal'},
        description="Opportunity fire",
        tactical_note="Snap shot during evasion - low probability but maintains pressure.",
        condition=Condition(
            condition_type=ConditionType.RANGE_BELOW,
            threshold=250.0,
            comparison="lt"
        )
    )

    # T+70s: Another break turn
    script.add_action(
        time_offset=70.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': BreakTurn,
            'turn_angle_deg': 45.0,
            'turn_direction': 'left',  # Opposite direction this time
            'burn_duration': 6.0,
            'throttle': 1.0
        },
        description="Break turn left",
        tactical_note="Unpredictable - alternate turn directions."
    )

    # T+85s: Resume jinking
    script.add_action(
        time_offset=85.0,
        action_type=ActionType.MANEUVER,
        parameters={
            'maneuver_class': EvasiveJink,
            'duration': 15.0,
            'jink_interval': 2.0,
            'throttle': 0.85
        },
        description="Resume evasive jinks",
        tactical_note="Continuous random motion - survival priority."
    )

    return script


# =============================================================================
# SCRIPT GENERATOR
# =============================================================================

class ScriptGenerator:
    """
    Generates varied combat scripts for testing.

    Creates randomized but tactically coherent scripts to provide
    diverse testing scenarios.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the script generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)

    def generate_random_script(
        self,
        duration: float = 120.0,
        aggressiveness: float = 0.5
    ) -> CombatScript:
        """
        Generate a randomized combat script.

        Args:
            duration: Total script duration in seconds
            aggressiveness: 0.0 (very defensive) to 1.0 (very aggressive)

        Returns:
            Generated CombatScript
        """
        script = CombatScript(
            name=f"Random_{self.rng.randint(1000, 9999)}",
            description=f"Randomly generated script (aggro: {aggressiveness:.1f})",
            total_duration=duration,
            repeat=self.rng.random() < 0.3,  # 30% chance to repeat
            base_aggressiveness=aggressiveness
        )

        # Number of actions based on duration
        num_actions = int(duration / 20) + self.rng.randint(1, 3)

        for i in range(num_actions):
            time_offset = (i / num_actions) * duration + self.rng.uniform(-5, 5)
            time_offset = max(0, min(duration - 5, time_offset))

            # Choose action type based on aggressiveness
            action = self._generate_random_action(time_offset, aggressiveness)
            script.actions.append(action)

        script.actions = sorted(script.actions, key=lambda a: a.time_offset)
        return script

    def _generate_random_action(
        self,
        time_offset: float,
        aggressiveness: float
    ) -> ScriptAction:
        """Generate a single random action."""

        # Action type probabilities based on aggressiveness
        if aggressiveness > 0.7:
            # Aggressive: more attacks
            action_weights = [
                (ActionType.MANEUVER, 0.4),
                (ActionType.FIRE_COILGUN, 0.3),
                (ActionType.LAUNCH_TORPEDO, 0.2),
                (ActionType.RETRACT_RADIATORS, 0.05),
                (ActionType.EXTEND_RADIATORS, 0.05),
            ]
        elif aggressiveness < 0.3:
            # Defensive: more evasion
            action_weights = [
                (ActionType.MANEUVER, 0.5),
                (ActionType.FIRE_COILGUN, 0.1),
                (ActionType.RETRACT_RADIATORS, 0.15),
                (ActionType.EXTEND_RADIATORS, 0.15),
                (ActionType.LAUNCH_TORPEDO, 0.1),
            ]
        else:
            # Balanced
            action_weights = [
                (ActionType.MANEUVER, 0.45),
                (ActionType.FIRE_COILGUN, 0.2),
                (ActionType.LAUNCH_TORPEDO, 0.15),
                (ActionType.EXTEND_RADIATORS, 0.1),
                (ActionType.RETRACT_RADIATORS, 0.1),
            ]

        # Weighted random choice
        total_weight = sum(w for _, w in action_weights)
        r = self.rng.random() * total_weight
        cumulative = 0
        action_type = ActionType.MANEUVER
        for at, weight in action_weights:
            cumulative += weight
            if r <= cumulative:
                action_type = at
                break

        # Generate parameters based on action type
        parameters = self._generate_action_parameters(action_type, aggressiveness)

        # Maybe add a condition
        condition = None
        if self.rng.random() < 0.3:  # 30% chance
            condition = self._generate_random_condition(aggressiveness)

        return ScriptAction(
            time_offset=time_offset,
            action_type=action_type,
            parameters=parameters,
            condition=condition,
            description=f"Random {action_type.name} action"
        )

    def _generate_action_parameters(
        self,
        action_type: ActionType,
        aggressiveness: float
    ) -> Dict[str, Any]:
        """Generate parameters for an action type."""

        if action_type == ActionType.MANEUVER:
            # Choose maneuver based on aggressiveness
            if aggressiveness > 0.6:
                maneuver_choices = [
                    (BurnToward, {'target_position': 'TARGET', 'throttle': 0.8 + self.rng.random() * 0.2}),
                    (SpiralApproach, {'target_position': 'TARGET', 'approach_throttle': 0.8}),
                ]
            elif aggressiveness < 0.4:
                maneuver_choices = [
                    (EvasiveJink, {'duration': 15 + self.rng.random() * 15, 'throttle': 0.7 + self.rng.random() * 0.3}),
                    (BurnAway, {'target_position': 'TARGET', 'throttle': 0.7}),
                    (RotateToBroadside, {'target_position': 'TARGET'}),
                ]
            else:
                maneuver_choices = [
                    (BurnToward, {'target_position': 'TARGET', 'throttle': 0.6}),
                    (EvasiveJink, {'duration': 10 + self.rng.random() * 10}),
                    (RotateToFace, {'target_position': 'TARGET'}),
                ]

            maneuver_class, params = self.rng.choice(maneuver_choices)
            params['maneuver_class'] = maneuver_class
            return params

        elif action_type == ActionType.FIRE_COILGUN:
            return {'weapon_slot': 'spinal'}

        elif action_type == ActionType.LAUNCH_TORPEDO:
            return {}

        return {}

    def _generate_random_condition(self, aggressiveness: float) -> Condition:
        """Generate a random condition."""
        if aggressiveness > 0.5:
            # Aggressive conditions: attack triggers
            condition_choices = [
                (ConditionType.RANGE_BELOW, 300.0, "lt"),
                (ConditionType.RANGE_BELOW, 500.0, "lt"),
            ]
        else:
            # Defensive conditions: survival triggers
            condition_choices = [
                (ConditionType.HEALTH_BELOW, 70.0, "lt"),
                (ConditionType.HEAT_ABOVE, 60.0, "gt"),
                (ConditionType.INCOMING_TORPEDO, 0.5, "gt"),
            ]

        ctype, threshold, comparison = self.rng.choice(condition_choices)
        return Condition(
            condition_type=ctype,
            threshold=threshold,
            comparison=comparison
        )

    def generate_reactive_script(self, threat_level: float = 0.5) -> CombatScript:
        """
        Generate a script that reacts to threat level.

        Higher threat = more defensive, more evasive.
        Lower threat = more aggressive, more offensive.

        Args:
            threat_level: 0.0 (no threat) to 1.0 (extreme threat)

        Returns:
            CombatScript tuned to threat level
        """
        # Inverse relationship: high threat = low aggression
        aggressiveness = 1.0 - threat_level

        script = CombatScript(
            name=f"Reactive_Threat{int(threat_level*100)}",
            description=f"Reactive script for threat level {threat_level:.2f}",
            total_duration=90.0,
            repeat=True,
            base_aggressiveness=aggressiveness
        )

        if threat_level > 0.7:
            # High threat: survival mode
            script.add_action(0.0, ActionType.RETRACT_RADIATORS,
                            description="Protect radiators from incoming fire")
            script.add_action(5.0, ActionType.MANEUVER,
                            parameters={'maneuver_class': EvasiveJink, 'duration': 30.0, 'throttle': 1.0},
                            description="Maximum evasive maneuvers")
            script.add_action(40.0, ActionType.MANEUVER,
                            parameters={'maneuver_class': BurnAway, 'target_position': 'TARGET', 'throttle': 1.0},
                            description="Escape burn")
        elif threat_level > 0.4:
            # Medium threat: balanced response
            script.add_action(0.0, ActionType.MANEUVER,
                            parameters={'maneuver_class': RotateToBroadside, 'target_position': 'TARGET'},
                            description="Present armor")
            script.add_action(20.0, ActionType.FIRE_COILGUN,
                            description="Return fire")
            script.add_action(40.0, ActionType.MANEUVER,
                            parameters={'maneuver_class': EvasiveJink, 'duration': 15.0},
                            description="Evasive while reloading")
            script.add_action(60.0, ActionType.FIRE_COILGUN,
                            description="Continue engagement")
        else:
            # Low threat: aggressive posture
            script.add_action(0.0, ActionType.EXTEND_RADIATORS,
                            description="Heat management for sustained operations")
            script.add_action(10.0, ActionType.MANEUVER,
                            parameters={'maneuver_class': BurnToward, 'target_position': 'TARGET', 'throttle': 1.0},
                            description="Close with target")
            script.add_action(30.0, ActionType.FIRE_COILGUN,
                            description="Engage at range")
            script.add_action(50.0, ActionType.LAUNCH_TORPEDO,
                            description="Torpedo salvo")
            script.add_action(60.0, ActionType.FIRE_COILGUN,
                            description="Follow-up salvo")

        return script


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_all_preset_scripts() -> Dict[str, CombatScript]:
    """
    Get all pre-built combat scripts.

    Returns:
        Dictionary mapping script name to CombatScript
    """
    return {
        'AggressiveApproach': create_aggressive_approach_script(),
        'DefensiveStance': create_defensive_stance_script(),
        'HitAndRun': create_hit_and_run_script(),
        'TorpedoRun': create_torpedo_run_script(),
        'EvasivePattern': create_evasive_pattern_script(),
    }


def create_script_for_situation(
    distance_km: float,
    own_health_percent: float,
    enemy_count: int,
    has_torpedoes: bool = True
) -> CombatScript:
    """
    Create a script appropriate for the tactical situation.

    Args:
        distance_km: Distance to nearest enemy
        own_health_percent: Our hull integrity percentage
        enemy_count: Number of enemy ships
        has_torpedoes: Whether we have torpedoes available

    Returns:
        Appropriate CombatScript for the situation
    """
    # Decision tree for script selection

    # Critical damage - survival mode
    if own_health_percent < 30:
        return create_evasive_pattern_script()

    # Multiple enemies - hit and run
    if enemy_count >= 2:
        return create_hit_and_run_script()

    # Long range with torpedoes - torpedo run
    if distance_km > 400 and has_torpedoes:
        return create_torpedo_run_script()

    # Close range and healthy - aggressive
    if distance_km < 200 and own_health_percent > 70:
        return create_aggressive_approach_script()

    # Default - defensive stance
    return create_defensive_stance_script()


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS COMBAT SCRIPTS - SELF TEST")
    print("=" * 70)

    # Test preset scripts
    print("\n--- Preset Combat Scripts ---")
    for name, script in get_all_preset_scripts().items():
        print(f"\n{name}:")
        print(f"  Description: {script.description}")
        print(f"  Duration: {script.total_duration}s")
        print(f"  Repeating: {script.repeat}")
        print(f"  Aggressiveness: {script.base_aggressiveness}")
        print(f"  Actions: {len(script.actions)}")

        # Show first few actions
        for i, action in enumerate(script.actions[:3]):
            print(f"    T+{action.time_offset:.0f}s: {action.description}")
            if action.tactical_note:
                print(f"           Tactic: {action.tactical_note[:60]}...")

    # Test script generator
    print("\n--- Script Generator ---")
    generator = ScriptGenerator(seed=42)

    random_script = generator.generate_random_script(
        duration=100.0,
        aggressiveness=0.7
    )
    print(f"\nRandom Script: {random_script.name}")
    print(f"  Actions: {len(random_script.actions)}")
    for action in random_script.actions[:5]:
        print(f"    T+{action.time_offset:.0f}s: {action.action_type.name}")

    # Test reactive scripts
    print("\n--- Reactive Scripts ---")
    for threat in [0.2, 0.5, 0.8]:
        reactive = generator.generate_reactive_script(threat_level=threat)
        print(f"\nThreat {threat:.1f}: {reactive.name}")
        print(f"  Aggressiveness: {reactive.base_aggressiveness:.1f}")
        for action in reactive.actions[:3]:
            print(f"    {action.description}")

    # Test situation-based script selection
    print("\n--- Situation-Based Selection ---")

    situations = [
        {"distance_km": 500, "own_health_percent": 90, "enemy_count": 1, "has_torpedoes": True},
        {"distance_km": 100, "own_health_percent": 40, "enemy_count": 2, "has_torpedoes": False},
        {"distance_km": 300, "own_health_percent": 20, "enemy_count": 1, "has_torpedoes": True},
    ]

    for sit in situations:
        script = create_script_for_situation(**sit)
        print(f"\nSituation: {sit}")
        print(f"  Selected: {script.name}")

    # Test executor (mock ship)
    print("\n--- Script Executor Test ---")

    # Create a mock ship for testing
    class MockShip:
        def __init__(self):
            self.hull_integrity = 80.0
            self.heat_percent = 30.0
            self.remaining_delta_v_kps = 5.0
            self.weapons = {}
            self.kinematic_state = None

    mock_ship = MockShip()
    aggressive_script = create_aggressive_approach_script()
    executor = ScriptExecutor(aggressive_script)

    # Start and run
    executor.start(current_time=0.0)
    print(f"Script started: {aggressive_script.name}")
    print(f"Is active: {executor.is_active}")

    # Simulate updates
    context = {
        'target_position': Vector3D(100000, 0, 0),
        'target_velocity': Vector3D(-500, 0, 0),
        'target_range_km': 100.0,
        'primary_target_id': 'enemy_1',
        'enemy_count': 1,
        'incoming_torpedoes': 0
    }

    for sim_time in [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]:
        results = executor.update(mock_ship, sim_time, context)
        if results:
            for result in results:
                status = "EXECUTED" if result.executed else "SKIPPED"
                print(f"  T+{sim_time:.0f}s: [{status}] {result.message}")

    print(f"\nFinal progress: {executor.progress:.1f}%")

    # Test condition evaluation
    print("\n--- Condition Evaluation ---")

    conditions = [
        Condition(ConditionType.HEALTH_BELOW, 50.0, "lt"),
        Condition(ConditionType.RANGE_BELOW, 200.0, "lt"),
        Condition(ConditionType.HEAT_ABOVE, 50.0, "gt"),
    ]

    for cond in conditions:
        result = cond.evaluate(mock_ship, context)
        print(f"  {cond.condition_type.name} ({cond.comparison} {cond.threshold}): {result}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
