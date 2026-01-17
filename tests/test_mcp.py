"""
Unit tests for MCP integration modules.

Tests:
- mcp_state.py - Shared state management
- mcp_chat.py - Admiral chat system
- mcp_controller.py - MCP controller
- fleet_config.py - MCPConfig parsing
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

# Import MCP modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.mcp_state import (
    MCPSharedState,
    MCPBattleState,
    MCPCommand,
    MCPCommandType,
    get_mcp_state,
)
from llm.mcp_chat import AdmiralChat, ChatMessage
from llm.mcp_controller import MCPController, MCPControllerConfig
from llm.fleet_config import BattleFleetConfig, MCPConfig


class TestMCPState:
    """Tests for MCPSharedState."""

    def setup_method(self):
        """Reset singleton before each test."""
        MCPSharedState.reset()

    def test_singleton(self):
        """Test that MCPSharedState is a singleton."""
        state1 = get_mcp_state()
        state2 = get_mcp_state()
        assert state1 is state2

    def test_register_faction(self):
        """Test faction registration."""
        state = get_mcp_state()
        state.register_faction("alpha")
        assert state.is_faction_active("alpha")
        assert not state.is_faction_active("beta")

    def test_unregister_faction(self):
        """Test faction unregistration."""
        state = get_mcp_state()
        state.register_faction("alpha")
        state.unregister_faction("alpha")
        assert not state.is_faction_active("alpha")

    def test_update_and_get_state(self):
        """Test state update and retrieval."""
        state = get_mcp_state()
        state.register_faction("alpha")

        battle_state = MCPBattleState(
            timestamp=100.0,
            faction="alpha",
            friendly_ships=[{"ship_id": "alpha_1"}],
            enemy_ships=[{"ship_id": "beta_1"}],
            is_battle_active=True,
        )
        state.update_state("alpha", battle_state)

        retrieved = state.get_state("alpha")
        assert retrieved.timestamp == 100.0
        assert retrieved.faction == "alpha"
        assert len(retrieved.friendly_ships) == 1

    def test_get_state_dict(self):
        """Test state dictionary conversion."""
        state = get_mcp_state()
        state.register_faction("alpha")

        battle_state = MCPBattleState(
            timestamp=50.0,
            faction="alpha",
            fleet_summary="2 ships",
        )
        state.update_state("alpha", battle_state)

        state_dict = state.get_state_dict("alpha")
        assert isinstance(state_dict, dict)
        assert state_dict["timestamp"] == 50.0
        assert state_dict["fleet_summary"] == "2 ships"

    def test_add_and_get_commands(self):
        """Test command queue."""
        state = get_mcp_state()
        state.register_faction("alpha")

        cmd = MCPCommand(
            command_type=MCPCommandType.SET_MANEUVER,
            ship_id="alpha_1",
            parameters={"maneuver_type": "INTERCEPT"},
        )
        state.add_command("alpha", cmd)

        commands = state.get_pending_commands("alpha")
        assert len(commands) == 1
        assert commands[0].command_type == MCPCommandType.SET_MANEUVER
        assert commands[0].ship_id == "alpha_1"

        # Commands should be cleared after get
        assert len(state.get_pending_commands("alpha")) == 0

    def test_peek_commands_does_not_clear(self):
        """Test that peek doesn't clear commands."""
        state = get_mcp_state()
        state.register_faction("alpha")

        cmd = MCPCommand(
            command_type=MCPCommandType.SET_PRIMARY_TARGET,
            ship_id="alpha_1",
            parameters={"target_id": "beta_1"},
        )
        state.add_command("alpha", cmd)

        # Peek should not clear
        peeked = state.peek_pending_commands("alpha")
        assert len(peeked) == 1

        # Should still be there
        peeked2 = state.peek_pending_commands("alpha")
        assert len(peeked2) == 1


class TestMCPBattleState:
    """Tests for MCPBattleState."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = MCPBattleState(
            timestamp=120.0,
            faction="alpha",
            friendly_ships=[{"ship_id": "alpha_1", "hull_integrity": 100}],
            enemy_ships=[{"ship_id": "beta_1"}],
            projectiles=[{"source": "alpha_1", "target": "beta_1"}],
            chat_history=[{"turn": 1, "content": "Hello"}],
            fleet_summary="1 Destroyer",
            is_battle_active=True,
            checkpoint_number=4,
            enemy_proposed_draw=False,
        )

        d = state.to_dict()
        assert d["timestamp"] == 120.0
        assert d["faction"] == "alpha"
        assert len(d["friendly_ships"]) == 1
        assert len(d["enemy_ships"]) == 1
        assert len(d["projectiles"]) == 1
        assert d["checkpoint_number"] == 4


class TestAdmiralChat:
    """Tests for AdmiralChat system."""

    def test_initial_state(self):
        """Test initial chat state."""
        chat = AdmiralChat()
        assert chat.can_send("alpha")
        assert chat.can_send("beta")
        assert chat.messages_remaining("alpha") == 3
        assert chat.get_current_turn() == 0

    def test_send_message_success(self):
        """Test successful message sending."""
        chat = AdmiralChat()
        result = chat.send_message("alpha", "Test message", 10.0)
        assert result is True
        assert chat.messages_remaining("alpha") == 2

    def test_send_message_limit(self):
        """Test message limit enforcement."""
        chat = AdmiralChat()

        # Send 3 messages (the limit)
        assert chat.send_message("alpha", "Message 1", 10.0)
        assert chat.send_message("alpha", "Message 2", 10.0)
        assert chat.send_message("alpha", "Message 3", 10.0)

        # Fourth message should fail
        result = chat.send_message("alpha", "Message 4", 10.0)
        assert result is False
        assert chat.messages_remaining("alpha") == 0
        assert not chat.can_send("alpha")

    def test_message_delivery_to_enemy(self):
        """Test that messages are queued for enemy delivery."""
        chat = AdmiralChat()
        chat.send_message("alpha", "Hello enemy", 10.0)

        # Alpha's message should be pending for beta
        pending = chat.get_pending_messages("beta")
        assert len(pending) == 1
        assert pending[0].content == "Hello enemy"
        assert pending[0].sender_faction == "alpha"

        # Should be cleared after get
        assert len(chat.get_pending_messages("beta")) == 0

    def test_peek_pending_does_not_clear(self):
        """Test that peek doesn't clear pending messages."""
        chat = AdmiralChat()
        chat.send_message("alpha", "Test", 10.0)

        peeked = chat.peek_pending_messages("beta")
        assert len(peeked) == 1

        peeked2 = chat.peek_pending_messages("beta")
        assert len(peeked2) == 1

    def test_new_turn_resets_counts(self):
        """Test that new_turn resets message counts."""
        chat = AdmiralChat()

        chat.send_message("alpha", "Turn 0", 10.0)
        chat.send_message("alpha", "Turn 0", 10.0)
        chat.send_message("alpha", "Turn 0", 10.0)
        assert not chat.can_send("alpha")

        chat.new_turn()
        assert chat.can_send("alpha")
        assert chat.messages_remaining("alpha") == 3

    def test_recent_history(self):
        """Test recent history retrieval."""
        chat = AdmiralChat(history_turns=5)

        chat.send_message("alpha", "Alpha message", 10.0)
        chat.send_message("beta", "Beta message", 11.0)

        # From alpha's perspective
        history = chat.get_recent_history("alpha")
        assert len(history) == 2
        assert history[0]["sender"] == "friendly"  # Alpha's own message
        assert history[1]["sender"] == "enemy"     # Beta's message

        # From beta's perspective
        history = chat.get_recent_history("beta")
        assert history[0]["sender"] == "enemy"     # Alpha's message
        assert history[1]["sender"] == "friendly"  # Beta's own message

    def test_history_turnlimit(self):
        """Test that history is limited by turns."""
        chat = AdmiralChat(history_turns=2)

        # Turn 0
        chat.send_message("alpha", "Turn 0", 0.0)
        chat.new_turn()

        # Turn 1
        chat.send_message("alpha", "Turn 1", 30.0)
        chat.new_turn()

        # Turn 2
        chat.send_message("alpha", "Turn 2", 60.0)
        chat.new_turn()

        # Turn 3
        chat.send_message("alpha", "Turn 3", 90.0)

        # Current turn is 3, history_turns=2 means min_turn = max(0, 3-2) = 1
        # So we see turns 1, 2, 3 (3 messages total)
        history = chat.get_recent_history("alpha")
        assert len(history) == 3
        assert history[0]["content"] == "Turn 1"
        assert history[1]["content"] == "Turn 2"
        assert history[2]["content"] == "Turn 3"

    def test_reset(self):
        """Test chat reset."""
        chat = AdmiralChat()
        chat.send_message("alpha", "Message", 10.0)
        chat.new_turn()
        chat.new_turn()

        chat.reset()
        assert chat.get_current_turn() == 0
        assert len(chat.get_all_history()) == 0
        assert chat.messages_remaining("alpha") == 3


class TestMCPController:
    """Tests for MCPController."""

    def test_init(self):
        """Test controller initialization."""
        config = MCPControllerConfig(
            faction="alpha",
            name="Test Commander",
        )
        controller = MCPController(
            config=config,
            fleet_data={"ships": {}},
        )

        assert controller.name == "Test Commander"
        assert controller.faction == "alpha"
        assert not controller.has_surrendered
        assert not controller.has_proposed_draw

    def test_set_ship_mapping(self):
        """Test ship name to ID mapping."""
        config = MCPControllerConfig(faction="alpha")
        controller = MCPController(config=config, fleet_data={})

        controller.set_ship_mapping({
            "TIS Alpha": "alpha_1",
            "TIS Beta": "alpha_2",
        })

        assert controller._ship_name_to_id["TIS Alpha"] == "alpha_1"
        assert controller._ship_id_to_name["alpha_1"] == "TIS Alpha"


class TestMCPConfigParsing:
    """Tests for MCPConfig in fleet_config.py."""

    def test_mcp_config_dataclass(self):
        """Test MCPConfig dataclass."""
        config = MCPConfig(
            enabled=True,
            transport="stdio",
            name="Test MCP",
            command_timeout=30.0,
        )

        assert config.enabled
        assert config.transport == "stdio"
        assert config.name == "Test MCP"
        assert config.command_timeout == 30.0

    def test_parse_fleet_with_mcp(self):
        """Test parsing fleet config with MCP."""
        config_data = {
            "battle_name": "MCP Test",
            "alpha_fleet": {
                "mcp": {
                    "enabled": True,
                    "name": "Alpha MCP",
                    "command_timeout": 45.0,
                },
                "ships": [
                    {"ship_id": "alpha_1", "ship_type": "destroyer", "model": "dummy"},
                ],
            },
            "beta_fleet": {
                "admiral": {
                    "model": "test/model",
                    "name": "Admiral Test",
                },
                "ships": [
                    {"ship_id": "beta_1", "ship_type": "frigate", "model": "test/model"},
                ],
            },
        }

        config = BattleFleetConfig.from_dict(config_data)

        assert config.alpha_fleet.mcp is not None
        assert config.alpha_fleet.mcp.enabled
        assert config.alpha_fleet.mcp.name == "Alpha MCP"
        assert config.alpha_fleet.mcp.command_timeout == 45.0
        assert config.alpha_fleet.admiral is None  # MCP takes precedence

        assert config.beta_fleet.mcp is None
        assert config.beta_fleet.admiral is not None

    def test_has_any_mcp(self):
        """Test has_any_mcp helper method."""
        config_data = {
            "battle_name": "Test",
            "alpha_fleet": {
                "mcp": {"enabled": True},
                "ships": [{"ship_type": "destroyer", "model": "dummy"}],
            },
            "beta_fleet": {
                "admiral": "test/model",
                "ships": [{"ship_type": "destroyer", "model": "test"}],
            },
        }

        config = BattleFleetConfig.from_dict(config_data)
        assert config.has_any_mcp()
        assert config.is_alpha_mcp()
        assert not config.is_beta_mcp()

    def test_mcp_disabled(self):
        """Test that disabled MCP is not parsed."""
        config_data = {
            "battle_name": "Test",
            "alpha_fleet": {
                "mcp": {"enabled": False},
                "admiral": "test/model",
                "ships": [{"ship_type": "destroyer", "model": "test"}],
            },
            "beta_fleet": {
                "ships": [{"ship_type": "destroyer", "model": "test"}],
            },
        }

        config = BattleFleetConfig.from_dict(config_data)
        assert config.alpha_fleet.mcp is None  # Disabled, so not parsed
        assert config.alpha_fleet.admiral is not None  # Admiral is used instead


class TestMCPCommandType:
    """Tests for MCPCommandType enum."""

    def test_all_command_types(self):
        """Test that all command types are defined."""
        assert MCPCommandType.SET_MANEUVER.value == "set_maneuver"
        assert MCPCommandType.SET_WEAPONS_ORDER.value == "set_weapons_order"
        assert MCPCommandType.SET_PRIMARY_TARGET.value == "set_primary_target"
        assert MCPCommandType.LAUNCH_TORPEDO.value == "launch_torpedo"
        assert MCPCommandType.SET_RADIATORS.value == "set_radiators"
        assert MCPCommandType.SEND_MESSAGE.value == "send_message"
        assert MCPCommandType.PROPOSE_DRAW.value == "propose_draw"
        assert MCPCommandType.SURRENDER.value == "surrender"
        assert MCPCommandType.READY.value == "ready"


class TestMCPCommand:
    """Tests for MCPCommand dataclass."""

    def test_create_command(self):
        """Test creating an MCP command."""
        cmd = MCPCommand(
            command_type=MCPCommandType.SET_MANEUVER,
            ship_id="alpha_1",
            parameters={"maneuver_type": "INTERCEPT", "throttle": 0.8},
            timestamp=100.0,
        )

        assert cmd.command_type == MCPCommandType.SET_MANEUVER
        assert cmd.ship_id == "alpha_1"
        assert cmd.parameters["maneuver_type"] == "INTERCEPT"
        assert cmd.parameters["throttle"] == 0.8
        assert cmd.timestamp == 100.0


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_to_dict(self):
        """Test ChatMessage serialization."""
        msg = ChatMessage(
            turn=3,
            timestamp=90.0,
            sender_faction="alpha",
            content="Test message",
            recipient="enemy",
        )

        d = msg.to_dict()
        assert d["turn"] == 3
        assert d["timestamp"] == 90.0
        assert d["content"] == "Test message"
        assert d["sender_faction"] == "alpha"


class TestBattlePlot:
    """Tests for generate_battle_plot function."""

    def test_velocity_arrow_directions(self):
        """Test velocity vector to arrow conversion."""
        from llm.mcp_server import get_velocity_arrow

        # Test all 8 cardinal directions
        assert get_velocity_arrow(1, 0) == "→"      # Right
        assert get_velocity_arrow(1, 1) == "↗"      # Up-right
        assert get_velocity_arrow(0, 1) == "↑"      # Up
        assert get_velocity_arrow(-1, 1) == "↖"    # Up-left
        assert get_velocity_arrow(-1, 0) == "←"    # Left
        assert get_velocity_arrow(-1, -1) == "↙"   # Down-left
        assert get_velocity_arrow(0, -1) == "↓"    # Down
        assert get_velocity_arrow(1, -1) == "↘"    # Down-right

        # Stationary
        assert get_velocity_arrow(0, 0) == "·"
        assert get_velocity_arrow(0.05, 0.05) == "·"  # Below threshold

    def test_generate_plot_basic(self):
        """Test basic plot generation."""
        from llm.mcp_server import generate_battle_plot

        state = {
            "timestamp": 180.0,
            "friendly_ships": [
                {
                    "ship_id": "alpha_1",
                    "ship_name": "Defiant",
                    "position_km": {"x": 0, "y": 0, "z": 0},
                    "velocity_vector": {"x": 1, "y": 0, "z": 0},
                    "hull_integrity": 93,
                },
            ],
            "enemy_ships": [
                {
                    "ship_id": "beta_1",
                    "ship_name": "Warbird",
                    "position_km": {"x": 100, "y": 0, "z": 0},
                    "velocity_vector": {"x": -1, "y": 0, "z": 0},
                },
            ],
        }

        plot = generate_battle_plot(state, "alpha", "xy")

        # Check basic structure
        assert "TACTICAL MAP" in plot
        assert "T=180s" in plot
        assert "[A1]" in plot  # Friendly ship label
        assert "[B1]" in plot  # Enemy ship label
        assert "Defiant" in plot
        assert "Warbird" in plot
        assert "(93%)" in plot  # Friendly hull shown
        assert "(enemy)" in plot  # Enemy marker (no hull - fog of war)
        assert "DISTANCES:" in plot
        assert "Scale:" in plot

    def test_generate_plot_no_ships(self):
        """Test plot with no ships."""
        from llm.mcp_server import generate_battle_plot

        state = {
            "timestamp": 0,
            "friendly_ships": [],
            "enemy_ships": [],
        }

        plot = generate_battle_plot(state, "alpha", "xy")
        assert plot == "No ships to display."

    def test_generate_plot_projections(self):
        """Test different projection planes."""
        from llm.mcp_server import generate_battle_plot

        state = {
            "timestamp": 60.0,
            "friendly_ships": [
                {
                    "ship_id": "alpha_1",
                    "ship_name": "Test Ship",
                    "position_km": {"x": 10, "y": 20, "z": 30},
                    "velocity_vector": {"x": 1, "y": 2, "z": 3},
                    "hull_integrity": 100,
                },
            ],
            "enemy_ships": [],
        }

        # Test XY projection
        plot_xy = generate_battle_plot(state, "alpha", "xy")
        assert "X/Y plane" in plot_xy

        # Test XZ projection
        plot_xz = generate_battle_plot(state, "alpha", "xz")
        assert "X/Z plane" in plot_xz

        # Test YZ projection
        plot_yz = generate_battle_plot(state, "alpha", "yz")
        assert "Y/Z plane" in plot_yz

    def test_generate_plot_faction_labels(self):
        """Test that faction labels are correct based on perspective."""
        from llm.mcp_server import generate_battle_plot

        state = {
            "timestamp": 0,
            "friendly_ships": [
                {
                    "ship_id": "alpha_1",
                    "ship_name": "Alpha Ship",
                    "position_km": {"x": 0, "y": 0, "z": 0},
                    "velocity_vector": {"x": 0, "y": 0, "z": 0},
                    "hull_integrity": 100,
                },
            ],
            "enemy_ships": [
                {
                    "ship_id": "beta_1",
                    "ship_name": "Beta Ship",
                    "position_km": {"x": 50, "y": 50, "z": 0},
                    "velocity_vector": {"x": 0, "y": 0, "z": 0},
                },
            ],
        }

        # From alpha's perspective: friendly=A, enemy=B
        plot_alpha = generate_battle_plot(state, "alpha", "xy")
        assert "[A1]" in plot_alpha
        assert "[B1]" in plot_alpha

        # From beta's perspective: friendly=B, enemy=A
        plot_beta = generate_battle_plot(state, "beta", "xy")
        assert "[B1]" in plot_beta
        assert "[A1]" in plot_beta

    def test_generate_plot_multiple_ships(self):
        """Test plot with multiple ships on each side."""
        from llm.mcp_server import generate_battle_plot

        state = {
            "timestamp": 120.0,
            "friendly_ships": [
                {
                    "ship_id": "alpha_1",
                    "ship_name": "Ship One",
                    "position_km": {"x": -50, "y": 0, "z": 0},
                    "velocity_vector": {"x": 1, "y": 0, "z": 0},
                    "hull_integrity": 100,
                },
                {
                    "ship_id": "alpha_2",
                    "ship_name": "Ship Two",
                    "position_km": {"x": -50, "y": 50, "z": 0},
                    "velocity_vector": {"x": 1, "y": -1, "z": 0},
                    "hull_integrity": 75,
                },
            ],
            "enemy_ships": [
                {
                    "ship_id": "beta_1",
                    "ship_name": "Enemy One",
                    "position_km": {"x": 50, "y": 0, "z": 0},
                    "velocity_vector": {"x": -1, "y": 0, "z": 0},
                },
                {
                    "ship_id": "beta_2",
                    "ship_name": "Enemy Two",
                    "position_km": {"x": 50, "y": -50, "z": 0},
                    "velocity_vector": {"x": -1, "y": 1, "z": 0},
                },
            ],
        }

        plot = generate_battle_plot(state, "alpha", "xy")

        # All ships should appear
        assert "[A1]" in plot
        assert "[A2]" in plot
        assert "[B1]" in plot
        assert "[B2]" in plot

        # Hull percentages for friendly only
        assert "(100%)" in plot
        assert "(75%)" in plot

        # Distances section should show multiple entries
        assert "[A1] →" in plot
        assert "[A2] →" in plot

    def test_generate_plot_distance_calculation(self):
        """Test that distances are calculated correctly."""
        from llm.mcp_server import generate_battle_plot

        state = {
            "timestamp": 0,
            "friendly_ships": [
                {
                    "ship_id": "alpha_1",
                    "ship_name": "Friendly",
                    "position_km": {"x": 0, "y": 0, "z": 0},
                    "velocity_vector": {"x": 0, "y": 0, "z": 0},
                    "hull_integrity": 100,
                },
            ],
            "enemy_ships": [
                {
                    "ship_id": "beta_1",
                    "ship_name": "Enemy",
                    "position_km": {"x": 30, "y": 40, "z": 0},  # 50km away (3-4-5 triangle)
                    "velocity_vector": {"x": 0, "y": 0, "z": 0},
                },
            ],
        }

        plot = generate_battle_plot(state, "alpha", "xy")

        # Distance should be 50km (sqrt(30^2 + 40^2) = 50)
        assert "50.0 km" in plot
