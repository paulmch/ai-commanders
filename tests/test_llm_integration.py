"""
Tests for LLM integration module.

Uses mocked LLM responses to test without API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import CaptainClient, ToolCall, LLMResponse
from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.prompts import CaptainPersonality, build_captain_prompt
from src.llm.communication import CommunicationChannel, CaptainMessage, MessageType
from src.llm.victory import VictoryEvaluator, BattleOutcome
from src.llm.tools import get_captain_tools, CAPTAIN_TOOLS


class TestCaptainTools:
    """Test tool definitions."""

    def test_base_tools_exist(self):
        """Base tools should be defined."""
        tools = get_captain_tools(has_torpedoes=False)
        tool_names = [t["function"]["name"] for t in tools]

        assert "set_maneuver" in tool_names
        assert "set_weapons_order" in tool_names
        assert "set_radiators" in tool_names
        assert "send_message" in tool_names
        assert "surrender" in tool_names
        assert "propose_draw" in tool_names
        assert "launch_torpedo" not in tool_names

    def test_torpedo_tool_conditional(self):
        """Torpedo tool should only appear when has_torpedoes=True."""
        tools_without = get_captain_tools(has_torpedoes=False)
        tools_with = get_captain_tools(has_torpedoes=True)

        names_without = [t["function"]["name"] for t in tools_without]
        names_with = [t["function"]["name"] for t in tools_with]

        assert "launch_torpedo" not in names_without
        assert "launch_torpedo" in names_with

    def test_maneuver_types(self):
        """Maneuver tool should have correct enum values."""
        tools = get_captain_tools()
        maneuver_tool = next(t for t in tools if t["function"]["name"] == "set_maneuver")

        params = maneuver_tool["function"]["parameters"]["properties"]
        maneuver_types = params["maneuver_type"]["enum"]

        assert "INTERCEPT" in maneuver_types
        assert "EVADE" in maneuver_types
        assert "BRAKE" in maneuver_types
        assert "MAINTAIN" in maneuver_types


class TestCommunicationChannel:
    """Test captain communication system."""

    def test_message_queue_and_delivery(self):
        """Messages should be queued and delivered to correct recipient."""
        channel = CommunicationChannel(
            alpha_name="Chen",
            alpha_ship="Relentless",
            beta_name="Volkov",
            beta_ship="Determination",
        )

        # Alpha sends message
        channel.queue_message("alpha", "Prepare to die!", 30.0)

        # Beta should receive it
        beta_messages = channel.deliver_messages("beta")
        assert len(beta_messages) == 1
        assert beta_messages[0].content == "Prepare to die!"
        assert beta_messages[0].sender_name == "Chen"

        # Alpha should not receive their own message
        alpha_messages = channel.deliver_messages("alpha")
        assert len(alpha_messages) == 0

    def test_surrender_detection(self):
        """Surrender should be tracked."""
        channel = CommunicationChannel(
            alpha_name="Chen",
            alpha_ship="Relentless",
            beta_name="Volkov",
            beta_ship="Determination",
        )

        assert not channel.has_surrender()
        assert not channel.is_battle_ended()

        channel.alpha_surrendered = True

        assert channel.has_surrender()
        assert channel.is_battle_ended()
        assert channel.get_surrender_loser() == "alpha"

    def test_mutual_draw(self):
        """Both captains must propose draw for it to succeed."""
        channel = CommunicationChannel(
            alpha_name="Chen",
            alpha_ship="Relentless",
            beta_name="Volkov",
            beta_ship="Determination",
        )

        assert not channel.has_mutual_draw()

        channel.alpha_proposed_draw = True
        assert not channel.has_mutual_draw()

        channel.beta_proposed_draw = True
        assert channel.has_mutual_draw()
        assert channel.is_battle_ended()


class TestVictoryEvaluator:
    """Test victory condition evaluation."""

    def test_destruction_victory(self):
        """Destroyed ship loses."""
        evaluator = VictoryEvaluator()

        alpha = Mock()
        alpha.is_destroyed = True
        alpha.hull_integrity = 0
        alpha.damage_dealt_gj = 50
        alpha.damage_taken_gj = 100

        beta = Mock()
        beta.is_destroyed = False
        beta.hull_integrity = 75
        beta.damage_dealt_gj = 100
        beta.damage_taken_gj = 50

        outcome, winner, reason = evaluator.evaluate(alpha, beta)

        assert outcome == BattleOutcome.BETA_VICTORY
        assert winner == "beta"
        assert "destroyed" in reason.lower()

    def test_mutual_destruction(self):
        """Both destroyed = draw."""
        evaluator = VictoryEvaluator()

        alpha = Mock()
        alpha.is_destroyed = True

        beta = Mock()
        beta.is_destroyed = True

        outcome, winner, reason = evaluator.evaluate(alpha, beta)

        assert outcome == BattleOutcome.DRAW
        assert winner is None
        assert "mutual" in reason.lower()

    def test_surrender_victory(self):
        """Surrender gives victory to other side."""
        evaluator = VictoryEvaluator()

        alpha = Mock()
        alpha.is_destroyed = False

        beta = Mock()
        beta.is_destroyed = False

        outcome, winner, reason = evaluator.evaluate(
            alpha, beta, alpha_surrendered=True
        )

        assert outcome == BattleOutcome.BETA_VICTORY
        assert winner == "beta"
        assert "surrender" in reason.lower()

    def test_damage_evaluation_at_time_limit(self):
        """At time limit, evaluate by damage ratio."""
        evaluator = VictoryEvaluator()

        # Alpha dealt more damage and has much more hull - clear advantage
        alpha = Mock()
        alpha.is_destroyed = False
        alpha.hull_integrity = 95  # Barely scratched
        alpha.damage_dealt_gj = 200  # Dealt lots of damage
        alpha.damage_taken_gj = 20
        alpha.module_layout = None
        # Need at least one operational weapon to avoid "disabled" state
        mock_weapon = Mock()
        mock_weapon.is_operational = True
        alpha.weapons = {"weapon_0": mock_weapon}

        beta = Mock()
        beta.is_destroyed = False
        beta.hull_integrity = 30  # Heavily damaged
        beta.damage_dealt_gj = 20
        beta.damage_taken_gj = 200
        beta.module_layout = None
        beta.weapons = {"weapon_0": mock_weapon}

        outcome, winner, reason = evaluator.evaluate(
            alpha, beta, at_time_limit=True
        )

        assert outcome == BattleOutcome.ALPHA_VICTORY
        assert winner == "alpha"


class TestPrompts:
    """Test prompt generation."""

    def test_build_captain_prompt(self):
        """Prompt should include all required sections."""
        ship_status = {
            "hull_integrity": 85,
            "heat_percent": 30,
            "delta_v_remaining": 450,
            "nose_armor": 10,
            "lateral_armor": 5,
            "tail_armor": 3,
            "heatsink_capacity": 525,
            "radiators_extended": False,
        }

        tactical_status = {
            "distance_km": 500,
            "closing_rate": 2.5,
            "enemy_bearing": "ahead",
            "threats": ["Torpedo 100km away"],
        }

        prompt = build_captain_prompt(
            captain_name="Chen",
            ship_name="Relentless",
            ship_status=ship_status,
            tactical_status=tactical_status,
            personality=CaptainPersonality.AGGRESSIVE,
        )

        # Check key sections exist
        assert "Captain Chen" in prompt
        assert "Relentless" in prompt
        assert "SIMULATION" in prompt
        assert "500" in prompt  # distance
        assert "85" in prompt  # hull
        assert "AGGRESSIVE" in prompt or "aggressive" in prompt.lower()
        assert "Spinal" in prompt or "spinal" in prompt.lower()  # Coilgun weapon


class TestLLMCaptain:
    """Test LLM captain decision-making."""

    def test_tool_execution_set_maneuver(self):
        """set_maneuver tool should create Maneuver command."""
        config = LLMCaptainConfig(
            name="Chen",
            ship_name="Relentless",
        )

        mock_client = Mock()
        captain = LLMCaptain(config, mock_client)

        # Create mock tool call
        tool_call = ToolCall(
            id="1",
            name="set_maneuver",
            arguments={"maneuver_type": "INTERCEPT", "throttle": 0.8},
        )

        # Create mock simulation
        mock_sim = Mock()
        mock_sim.current_time = 30.0
        # Mock enemy for INTERCEPT target_id
        mock_enemy = Mock()
        mock_enemy.ship_id = "beta"
        mock_sim.get_enemy_ships.return_value = [mock_enemy]

        cmd = captain._execute_tool(tool_call, mock_sim, "alpha")

        assert cmd is not None
        assert cmd.throttle == 0.8

    def test_tool_execution_send_message(self):
        """send_message tool should queue message, not return command."""
        config = LLMCaptainConfig(
            name="Chen",
            ship_name="Relentless",
        )

        mock_client = Mock()
        captain = LLMCaptain(config, mock_client)

        tool_call = ToolCall(
            id="1",
            name="send_message",
            arguments={"message": "Prepare to be destroyed!"},
        )

        mock_sim = Mock()

        cmd = captain._execute_tool(tool_call, mock_sim, "alpha")

        assert cmd is None  # No command returned
        assert captain.pending_message == "Prepare to be destroyed!"

    def test_tool_execution_surrender(self):
        """surrender tool should set flag."""
        config = LLMCaptainConfig(
            name="Chen",
            ship_name="Relentless",
        )

        mock_client = Mock()
        captain = LLMCaptain(config, mock_client)

        tool_call = ToolCall(
            id="1",
            name="surrender",
            arguments={},
        )

        mock_sim = Mock()

        assert not captain.has_surrendered

        cmd = captain._execute_tool(tool_call, mock_sim, "alpha")

        assert cmd is None
        assert captain.has_surrendered


class TestIntegration:
    """Integration tests with mocked LLM."""

    def test_captain_makes_decisions(self):
        """Captain should make decisions via tool calls."""
        from src.physics import Vector3D

        # Create mock client that returns tool calls directly
        mock_client = Mock()
        mock_client.decide_with_tools.return_value = [
            ToolCall(
                id="call_1",
                name="set_maneuver",
                arguments={"maneuver_type": "INTERCEPT", "throttle": 1.0}
            )
        ]

        config = LLMCaptainConfig(name="Chen", ship_name="Relentless")
        captain = LLMCaptain(config, mock_client)

        # Mock simulation with real Vector3D for positions/velocities
        mock_ship = Mock()
        mock_ship.is_destroyed = False
        mock_ship.hull_integrity = 100
        mock_ship.remaining_delta_v_kps = 500
        mock_ship.thermal_system = Mock()
        mock_ship.thermal_system.heat_percent = 20
        mock_ship.thermal_system.heatsink = Mock()
        mock_ship.thermal_system.heatsink.capacity_gj = 525
        mock_ship.thermal_system.radiators = None
        mock_ship.armor = None
        mock_ship.position = Vector3D(-250000, 0, 0)  # Real Vector3D
        mock_ship.velocity = Vector3D(0, 0, 0)
        mock_ship.forward = Vector3D(1, 0, 0)  # Pointing at enemy
        mock_ship.shots_fired = 0
        mock_ship.hits_scored = 0
        mock_ship.damage_dealt_gj = 0.0
        mock_ship.damage_taken_gj = 0.0
        mock_ship.ship_id = "alpha"
        mock_ship.module_layout = None  # No module layout for simplicity

        # Mock weapons dict
        mock_weapon = Mock()
        mock_weapon.is_operational = True
        mock_weapon.is_ready = True
        mock_weapon.cooldown_remaining = 0
        mock_ship.weapons = {"spinal": mock_weapon, "turret": mock_weapon}

        mock_enemy = Mock()
        mock_enemy.ship_id = "beta"
        mock_enemy.position = Vector3D(250000, 1000, 0)  # Real Vector3D
        mock_enemy.velocity = Vector3D(0, 0, 0)
        mock_enemy.armor = None  # No armor for simplicity
        mock_enemy.hull_integrity = 100  # Hull damage info
        mock_enemy.shots_fired = 0
        mock_enemy.hits_scored = 0
        mock_enemy.damage_dealt_gj = 0.0
        mock_enemy.damage_taken_gj = 0.0

        mock_sim = Mock()
        mock_sim.current_time = 30.0
        mock_sim.get_ship = Mock(return_value=mock_ship)
        mock_sim.get_enemy_ships = Mock(return_value=[mock_enemy])
        mock_sim.torpedoes = []  # Empty torpedoes list
        mock_sim.projectiles = []  # Empty projectiles list

        # Make decision
        commands = captain.decide("alpha", mock_sim)

        assert len(commands) == 1
        assert captain.decision_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
