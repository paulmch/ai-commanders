"""
Victory condition evaluation for LLM battles.

Determines battle outcome based on ship destruction,
surrender, draw agreement, or damage comparison at time limit.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
from enum import Enum


class BattleOutcome(Enum):
    """Possible battle outcomes."""
    ALPHA_VICTORY = "alpha_victory"
    BETA_VICTORY = "beta_victory"
    DRAW = "draw"
    ONGOING = "ongoing"


class VictoryEvaluator:
    """
    Evaluates victory conditions for LLM battles.

    Victory can occur via:
    1. Ship destruction (hull integrity <= 0 or critical module destroyed)
    2. Surrender (captain choice)
    3. Mutual draw agreement
    4. Time/checkpoint limit + damage evaluation
    """

    def is_ship_destroyed(self, ship: Any) -> bool:
        """Check if ship is destroyed."""
        return ship.is_destroyed

    def is_ship_disabled(self, ship: Any) -> bool:
        """
        Check if ship is combat-ineffective.

        A ship is disabled if:
        - Critical module (bridge/reactor) destroyed
        - All weapons destroyed
        - Engine destroyed (no maneuverability)
        """
        if ship.module_layout:
            for module in ship.module_layout.get_all_modules():
                if module.is_critical and module.is_destroyed:
                    return True

        # Check if any weapons are still operational
        if hasattr(ship, 'weapons'):
            weapons_operational = any(
                w.is_operational for w in ship.weapons.values()
            )
            if not weapons_operational:
                return True

        return False

    def evaluate_by_damage(
        self,
        alpha: Any,
        beta: Any,
    ) -> Tuple[BattleOutcome, Optional[str], str]:
        """
        Evaluate winner based on damage dealt vs received.

        Used when battle ends via time limit without destruction.

        Args:
            alpha: Alpha ship state
            beta: Beta ship state

        Returns:
            Tuple of (outcome, winner_id, reason)
        """
        # Calculate damage efficiency ratio
        alpha_dealt = alpha.damage_dealt_gj
        alpha_taken = max(1.0, alpha.damage_taken_gj)
        beta_dealt = beta.damage_dealt_gj
        beta_taken = max(1.0, beta.damage_taken_gj)

        alpha_ratio = alpha_dealt / alpha_taken
        beta_ratio = beta_dealt / beta_taken

        # Also consider hull integrity
        alpha_hull = alpha.hull_integrity
        beta_hull = beta.hull_integrity

        # Combined score: 40% damage ratio, 60% hull remaining
        alpha_score = (alpha_ratio * 40) + (alpha_hull * 0.6)
        beta_score = (beta_ratio * 40) + (beta_hull * 0.6)

        # Margin for draw
        margin = 5.0

        if abs(alpha_score - beta_score) < margin:
            return (
                BattleOutcome.DRAW,
                None,
                f"Too close to call (Alpha: {alpha_score:.1f} vs Beta: {beta_score:.1f})"
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

    def evaluate(
        self,
        alpha: Any,
        beta: Any,
        alpha_surrendered: bool = False,
        beta_surrendered: bool = False,
        mutual_draw: bool = False,
        at_time_limit: bool = False,
    ) -> Tuple[BattleOutcome, Optional[str], str]:
        """
        Comprehensive victory evaluation.

        Args:
            alpha: Alpha ship state
            beta: Beta ship state
            alpha_surrendered: True if alpha captain surrendered
            beta_surrendered: True if beta captain surrendered
            mutual_draw: True if both captains agreed to draw
            at_time_limit: True if battle reached time/checkpoint limit

        Returns:
            Tuple of (outcome, winner_id, reason)
        """
        # Check surrender first
        if alpha_surrendered:
            return (BattleOutcome.BETA_VICTORY, "beta", "Alpha surrendered")
        if beta_surrendered:
            return (BattleOutcome.ALPHA_VICTORY, "alpha", "Beta surrendered")

        # Check mutual draw
        if mutual_draw:
            return (BattleOutcome.DRAW, None, "Mutual draw agreed")

        # Check destruction
        alpha_destroyed = self.is_ship_destroyed(alpha)
        beta_destroyed = self.is_ship_destroyed(beta)

        if alpha_destroyed and beta_destroyed:
            return (BattleOutcome.DRAW, None, "Mutual destruction")
        elif alpha_destroyed:
            return (BattleOutcome.BETA_VICTORY, "beta", "Alpha destroyed")
        elif beta_destroyed:
            return (BattleOutcome.ALPHA_VICTORY, "alpha", "Beta destroyed")

        # Check disabled
        alpha_disabled = self.is_ship_disabled(alpha)
        beta_disabled = self.is_ship_disabled(beta)

        if alpha_disabled and beta_disabled:
            return (BattleOutcome.DRAW, None, "Both ships disabled")
        elif alpha_disabled:
            return (BattleOutcome.BETA_VICTORY, "beta", "Alpha disabled")
        elif beta_disabled:
            return (BattleOutcome.ALPHA_VICTORY, "alpha", "Beta disabled")

        # At time limit, evaluate by damage
        if at_time_limit:
            return self.evaluate_by_damage(alpha, beta)

        # Battle ongoing
        return (BattleOutcome.ONGOING, None, "Battle continues")
