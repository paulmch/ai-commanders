"""
Comprehensive test suite for the BattleRecorder module.

Tests cover:
- Single ship (1v1) battle recording
- Multi-ship fleet battle recording
- Sim trace recording with ship positions, velocities, hull%, destroyed status
- Module damage events
- Hit/miss events
- Conversation/message events
- JSON serialization

Run with: uv run pytest tests/test_battle_recorder.py -v
"""

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock

import pytest

from src.llm.battle_recorder import (
    BattleEvent,
    BattleRecording,
    BattleRecorder,
    EventType,
    create_battle_filename,
)


# =============================================================================
# Fixtures - Mock configurations and ships
# =============================================================================

class MockPersonality(str, Enum):
    """Mock personality enum for testing."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    DEFENSIVE = "defensive"


@pytest.fixture
def mock_captain_config():
    """Create a mock captain configuration."""
    def _create(
        name: str = "Captain Test",
        model: str = "test-model/v1",
        ship_name: str = "TIS Test Ship",
        personality: MockPersonality = MockPersonality.BALANCED
    ):
        config = Mock()
        config.name = name
        config.model = model
        config.ship_name = ship_name
        config.personality = personality
        return config
    return _create


@pytest.fixture
def mock_battle_config():
    """Create a mock battle configuration."""
    config = Mock()
    config.initial_distance_km = 500.0
    config.time_limit_s = 1200.0
    config.max_checkpoints = 10
    config.unlimited_mode = False
    return config


@pytest.fixture
def mock_armor():
    """Create a mock armor system."""
    armor = Mock()
    nose_section = Mock()
    nose_section.thickness_cm = 150.0
    nose_section.material = "titanium"

    lateral_section = Mock()
    lateral_section.thickness_cm = 25.0
    lateral_section.material = "titanium"

    tail_section = Mock()
    tail_section.thickness_cm = 30.0
    tail_section.material = "titanium"

    def get_section(name):
        sections = {
            "nose": nose_section,
            "lateral": lateral_section,
            "tail": tail_section,
        }
        return sections.get(name)

    armor.get_section = get_section
    return armor


@pytest.fixture
def mock_ship(mock_armor):
    """Create a mock ship with all required attributes."""
    def _create(
        ship_id: str = "alpha",
        name: str = "TIS Test Ship",
        max_accel_g: float = 2.0,
        delta_v_kps: float = 500.0,
    ):
        ship = Mock()
        ship.ship_id = ship_id
        ship.name = name
        ship.armor = mock_armor
        ship.max_acceleration_g = max_accel_g
        ship.delta_v_budget_kps = delta_v_kps

        # Weapons
        coilgun = Mock()
        coilgun.damage_gj = 2.5
        type(coilgun).__name__ = "Coilgun"

        laser = Mock()
        laser.damage_gj = 1.0
        type(laser).__name__ = "Laser"

        ship.weapons = {
            "weapon_0": coilgun,
            "weapon_1": laser,
        }

        return ship
    return _create


@pytest.fixture
def mock_battle_result():
    """Create a mock battle result."""
    class MockOutcome(str, Enum):
        ALPHA_VICTORY = "alpha_victory"
        BETA_VICTORY = "beta_victory"
        DRAW = "draw"

    result = Mock()
    result.winner = "alpha"
    result.reason = "Enemy ship destroyed"
    result.outcome = MockOutcome.ALPHA_VICTORY
    result.duration_s = 450.0
    result.checkpoints_used = 5
    return result


@pytest.fixture
def mock_fleet_config():
    """Create a mock fleet battle configuration."""
    fleet_config = Mock()
    fleet_config.battle_name = "Test Fleet Battle"
    fleet_config.initial_distance_km = 1000.0
    fleet_config.time_limit_s = 1800.0
    fleet_config.max_checkpoints = 20

    # Alpha fleet
    alpha_fleet = Mock()
    alpha_ship_config = Mock()
    alpha_ship_config.model = "alpha-model/v1"
    alpha_fleet.ships = [alpha_ship_config]
    fleet_config.alpha_fleet = alpha_fleet

    # Beta fleet
    beta_fleet = Mock()
    beta_ship_config = Mock()
    beta_ship_config.model = "beta-model/v1"
    beta_fleet.ships = [beta_ship_config]
    fleet_config.beta_fleet = beta_fleet

    return fleet_config


@pytest.fixture
def mock_admiral():
    """Create a mock admiral for fleet battles."""
    def _create(name: str = "Admiral Test", model: str = "admiral-model/v1"):
        admiral = Mock()
        admiral.name = name
        admiral.config = Mock()
        admiral.config.model = model
        return admiral
    return _create


# =============================================================================
# Test BattleEvent
# =============================================================================

class TestBattleEvent:
    """Tests for the BattleEvent dataclass."""

    def test_basic_event_creation(self):
        """Test creating a basic battle event."""
        event = BattleEvent(
            timestamp=10.5,
            event_type=EventType.SHOT_FIRED,
            ship_id="alpha",
            data={"target_id": "beta", "damage_gj": 2.5}
        )

        assert event.timestamp == 10.5
        assert event.event_type == EventType.SHOT_FIRED
        assert event.ship_id == "alpha"
        assert event.data["target_id"] == "beta"
        assert event.data["damage_gj"] == 2.5

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = BattleEvent(
            timestamp=100.0,
            event_type=EventType.HIT,
            ship_id="beta",
            data={
                "shooter_id": "alpha",
                "hit_location": "nose",
                "damage_gj": 5.0,
            }
        )

        result = event.to_dict()

        assert result["timestamp"] == 100.0
        assert result["event_type"] == EventType.HIT
        assert result["ship_id"] == "beta"
        assert result["data"]["shooter_id"] == "alpha"
        assert result["data"]["hit_location"] == "nose"

    def test_event_with_no_ship_id(self):
        """Test creating event without ship_id (e.g., battle_start)."""
        event = BattleEvent(
            timestamp=0.0,
            event_type=EventType.BATTLE_START,
            data={"distance_km": 500.0}
        )

        assert event.ship_id is None
        assert event.data["distance_km"] == 500.0

    def test_event_default_data(self):
        """Test that data defaults to empty dict."""
        event = BattleEvent(
            timestamp=0.0,
            event_type=EventType.CHECKPOINT,
        )

        assert event.data == {}


# =============================================================================
# Test BattleRecording
# =============================================================================

class TestBattleRecording:
    """Tests for the BattleRecording dataclass."""

    def test_default_recording(self):
        """Test creating a recording with defaults."""
        recording = BattleRecording()

        assert recording.recording_version == "2.1"
        assert recording.is_fleet_battle is False
        assert recording.events == []
        assert recording.sim_trace == []
        assert recording.winner is None

    def test_recording_to_dict(self):
        """Test converting recording to dictionary."""
        recording = BattleRecording(
            alpha_model="model-a",
            beta_model="model-b",
            alpha_name="Captain A",
            beta_name="Captain B",
            initial_distance_km=500.0,
        )

        result = recording.to_dict()

        assert result["alpha_model"] == "model-a"
        assert result["beta_model"] == "model-b"
        assert result["alpha_name"] == "Captain A"
        assert result["beta_name"] == "Captain B"
        assert result["initial_distance_km"] == 500.0

    def test_recording_to_json(self):
        """Test JSON serialization."""
        recording = BattleRecording(
            alpha_model="model-a",
            beta_model="model-b",
            winner="alpha",
            result_reason="Victory",
        )

        json_str = recording.to_json()
        parsed = json.loads(json_str)

        assert parsed["alpha_model"] == "model-a"
        assert parsed["beta_model"] == "model-b"
        assert parsed["winner"] == "alpha"
        assert parsed["result_reason"] == "Victory"

    def test_recording_with_fleet_battle_fields(self):
        """Test recording with fleet battle specific fields."""
        recording = BattleRecording(
            is_fleet_battle=True,
            battle_name="Epic Fleet Battle",
            alpha_fleet={"ships": [{"ship_id": "alpha_1"}]},
            beta_fleet={"ships": [{"ship_id": "beta_1"}]},
            alpha_ships_remaining=2,
            beta_ships_remaining=1,
        )

        result = recording.to_dict()

        assert result["is_fleet_battle"] is True
        assert result["battle_name"] == "Epic Fleet Battle"
        assert len(result["alpha_fleet"]["ships"]) == 1
        assert result["alpha_ships_remaining"] == 2


# =============================================================================
# Test BattleRecorder - Single Ship (1v1) Recording
# =============================================================================

class TestBattleRecorder1v1:
    """Tests for single ship (1v1) battle recording."""

    def test_start_recording(self, mock_battle_config, mock_captain_config, mock_ship):
        """Test starting a 1v1 recording."""
        recorder = BattleRecorder()
        alpha_config = mock_captain_config(name="Captain Alpha", model="alpha/v1")
        beta_config = mock_captain_config(name="Captain Beta", model="beta/v1")
        alpha_ship = mock_ship(ship_id="alpha")
        beta_ship = mock_ship(ship_id="beta")

        recorder.start_recording(
            mock_battle_config,
            alpha_config,
            beta_config,
            alpha_ship,
            beta_ship,
        )

        recording = recorder.get_recording()

        assert recorder._is_recording is True
        assert recording.alpha_model == "alpha/v1"
        assert recording.beta_model == "beta/v1"
        assert recording.alpha_name == "Captain Alpha"
        assert recording.beta_name == "Captain Beta"
        assert recording.initial_distance_km == 500.0
        assert recording.time_limit_s == 1200.0
        assert recording.max_checkpoints == 10
        assert recording.is_fleet_battle is False

        # Verify ship specs extracted
        assert recording.alpha_specs["ship_id"] == "alpha"
        assert recording.beta_specs["ship_id"] == "beta"

        # Verify BATTLE_START event recorded
        assert len(recorder.events) == 1
        assert recorder.events[0].event_type == EventType.BATTLE_START

    def test_record_checkpoint(self, mock_battle_config, mock_captain_config, mock_ship):
        """Test recording a checkpoint event."""
        recorder = BattleRecorder()
        alpha_config = mock_captain_config()
        beta_config = mock_captain_config()

        recorder.start_recording(mock_battle_config, alpha_config, beta_config)

        alpha_state = {
            "ship_id": "alpha",
            "position": [0, 0, 0],
            "velocity": [100, 0, 0],
            "hull_integrity": 85.0,
            "is_destroyed": False,
        }
        beta_state = {
            "ship_id": "beta",
            "position": [500000, 0, 0],
            "velocity": [-100, 0, 0],
            "hull_integrity": 90.0,
            "is_destroyed": False,
        }

        recorder.record_checkpoint(
            timestamp=30.0,
            checkpoint_num=1,
            alpha_state=alpha_state,
            beta_state=beta_state,
            distance_km=500.0,
        )

        # BATTLE_START + CHECKPOINT
        assert len(recorder.events) == 2
        checkpoint_event = recorder.events[1]
        assert checkpoint_event.event_type == EventType.CHECKPOINT
        assert checkpoint_event.timestamp == 30.0
        assert checkpoint_event.data["checkpoint"] == 1
        assert checkpoint_event.data["distance_km"] == 500.0
        assert checkpoint_event.data["alpha"]["hull_integrity"] == 85.0

    def test_record_all_event_types(self, mock_battle_config, mock_captain_config):
        """Test that all event types can be recorded in a 1v1 battle."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(name="Alpha"),
            mock_captain_config(name="Beta"),
        )

        # Record various events
        recorder.record_maneuver(10.0, "alpha", "INTERCEPT", 1.0, "beta")
        recorder.record_weapons_order(15.0, "alpha", "weapon_0", "FIRE_AT_WILL", "beta")
        recorder.record_shot_fired(20.0, "alpha", "beta", "weapon_0", 0.75, 450.0, 2.5, 8.0)
        recorder.record_hit(
            25.0, "alpha", "beta", "weapon_0", "nose", 15.0, 2.5, 1.0, 149.0, 1.5, False
        )
        recorder.record_miss(30.0, "alpha", "beta", "weapon_0", 0.3, 400.0)
        recorder.record_armor_damage(35.0, "beta", "nose", 1.0, 149.0, 0.1)
        recorder.record_module_damaged(40.0, "beta", "engine_1", 0.5, False)
        recorder.record_module_destroyed(45.0, "beta", "radiator_1")
        recorder.record_message(50.0, "alpha", "Captain Alpha", "TIS Alpha", "Prepare to meet your maker!")
        recorder.record_surrender(55.0, "beta", "Captain Beta")
        recorder.record_draw_proposal(60.0, "alpha", "Captain Alpha")
        recorder.record_radiator_change(65.0, "alpha", True)
        recorder.record_thermal_warning(70.0, "beta", 85.0, False)

        # Verify event count (BATTLE_START + 13 other events)
        assert len(recorder.events) == 14

        # Verify event types
        event_types = [e.event_type for e in recorder.events]
        assert EventType.BATTLE_START in event_types
        assert EventType.MANEUVER in event_types
        assert EventType.WEAPONS_ORDER in event_types
        assert EventType.SHOT_FIRED in event_types
        assert EventType.HIT in event_types
        assert EventType.MISS in event_types
        assert EventType.ARMOR_DAMAGE in event_types
        assert EventType.MODULE_DAMAGED in event_types
        assert EventType.MODULE_DESTROYED in event_types
        assert EventType.MESSAGE in event_types
        assert EventType.SURRENDER in event_types
        assert EventType.DRAW_PROPOSAL in event_types
        assert EventType.RADIATOR_CHANGE in event_types
        assert EventType.THERMAL_WARNING in event_types

    def test_end_recording(self, mock_battle_config, mock_captain_config, mock_battle_result):
        """Test ending a recording properly."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.end_recording(mock_battle_result, final_time=450.0)

        recording = recorder.get_recording()

        assert recorder._is_recording is False
        assert recording.winner == "alpha"
        assert recording.result_reason == "Enemy ship destroyed"
        assert recording.duration_s == 450.0
        assert recording.total_checkpoints == 5

        # Verify BATTLE_END event
        end_events = [e for e in recording.events if e["event_type"] == EventType.BATTLE_END]
        assert len(end_events) == 1
        assert end_events[0]["data"]["winner"] == "alpha"


# =============================================================================
# Test BattleRecorder - Multi-Ship Fleet Battle Recording
# =============================================================================

class TestBattleRecorderFleet:
    """Tests for multi-ship fleet battle recording."""

    def test_start_fleet_recording(
        self, mock_fleet_config, mock_battle_config, mock_ship, mock_admiral
    ):
        """Test starting a fleet battle recording."""
        recorder = BattleRecorder()

        alpha_ships = {
            "alpha_1": mock_ship(ship_id="alpha_1", name="TIS Alpha One"),
            "alpha_2": mock_ship(ship_id="alpha_2", name="TIS Alpha Two"),
        }
        beta_ships = {
            "beta_1": mock_ship(ship_id="beta_1", name="HFS Beta One"),
            "beta_2": mock_ship(ship_id="beta_2", name="HFS Beta Two"),
        }
        alpha_admiral = mock_admiral(name="Admiral Alpha", model="admiral-alpha/v1")
        beta_admiral = mock_admiral(name="Admiral Beta", model="admiral-beta/v1")

        recorder.start_fleet_recording(
            mock_fleet_config,
            mock_battle_config,
            alpha_ships,
            beta_ships,
            alpha_admiral,
            beta_admiral,
        )

        recording = recorder.get_recording()

        assert recorder._is_recording is True
        assert recording.is_fleet_battle is True
        assert recording.battle_name == "Test Fleet Battle"
        assert recording.initial_distance_km == 1000.0

        # Verify fleet data
        assert recording.alpha_fleet["admiral"]["name"] == "Admiral Alpha"
        assert recording.alpha_fleet["admiral"]["model"] == "admiral-alpha/v1"
        assert len(recording.alpha_fleet["ships"]) == 2

        assert recording.beta_fleet["admiral"]["name"] == "Admiral Beta"
        assert len(recording.beta_fleet["ships"]) == 2

        # Verify BATTLE_START event has fleet info
        start_event = recorder.events[0]
        assert start_event.event_type == EventType.BATTLE_START
        assert start_event.data["is_fleet_battle"] is True
        assert start_event.data["alpha_ships"] == 2
        assert start_event.data["beta_ships"] == 2

    def test_fleet_all_ship_ids_captured(
        self, mock_fleet_config, mock_battle_config, mock_ship, mock_admiral
    ):
        """Test that all ship IDs are captured in fleet recordings."""
        recorder = BattleRecorder()

        # Create multiple ships per side
        alpha_ships = {
            f"alpha_{i}": mock_ship(ship_id=f"alpha_{i}", name=f"Alpha Ship {i}")
            for i in range(1, 4)
        }
        beta_ships = {
            f"beta_{i}": mock_ship(ship_id=f"beta_{i}", name=f"Beta Ship {i}")
            for i in range(1, 5)
        }

        recorder.start_fleet_recording(
            mock_fleet_config,
            mock_battle_config,
            alpha_ships,
            beta_ships,
            mock_admiral(),
            mock_admiral(),
        )

        recording = recorder.get_recording()

        # Verify all ships are recorded
        alpha_ship_ids = [s["ship_id"] for s in recording.alpha_fleet["ships"]]
        beta_ship_ids = [s["ship_id"] for s in recording.beta_fleet["ships"]]

        assert "alpha_1" in alpha_ship_ids
        assert "alpha_2" in alpha_ship_ids
        assert "alpha_3" in alpha_ship_ids
        assert len(alpha_ship_ids) == 3

        assert "beta_1" in beta_ship_ids
        assert "beta_2" in beta_ship_ids
        assert "beta_3" in beta_ship_ids
        assert "beta_4" in beta_ship_ids
        assert len(beta_ship_ids) == 4

    def test_fleet_battle_without_admirals(
        self, mock_fleet_config, mock_battle_config, mock_ship
    ):
        """Test fleet recording without admirals (captain-only mode)."""
        recorder = BattleRecorder()

        alpha_ships = {"alpha_1": mock_ship(ship_id="alpha_1")}
        beta_ships = {"beta_1": mock_ship(ship_id="beta_1")}

        recorder.start_fleet_recording(
            mock_fleet_config,
            mock_battle_config,
            alpha_ships,
            beta_ships,
            alpha_admiral=None,
            beta_admiral=None,
        )

        recording = recorder.get_recording()

        assert recording.is_fleet_battle is True
        assert recording.alpha_fleet["admiral"] is None
        assert recording.beta_fleet["admiral"] is None
        assert len(recording.alpha_fleet["ships"]) == 1


# =============================================================================
# Test Sim Trace Recording
# =============================================================================

class TestSimTraceRecording:
    """Tests for simulation trace recording (per-frame state capture)."""

    def test_record_sim_frame_basic(self, mock_battle_config, mock_captain_config):
        """Test recording a basic simulation frame."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        ships = {
            "alpha": {
                "position": (0.0, 0.0, 0.0),
                "velocity": (100.0, 0.0, 0.0),
                "forward": (1.0, 0.0, 0.0),
                "thrust": 0.5,
                "maneuver": "INTERCEPT",
                "is_destroyed": False,
                "hull_pct": 95.0,
            },
            "beta": {
                "position": (500000.0, 1000.0, 0.0),
                "velocity": (-100.0, 0.0, 0.0),
                "forward": (-1.0, 0.0, 0.0),
                "thrust": 0.3,
                "maneuver": "EVADE",
                "is_destroyed": False,
                "hull_pct": 80.0,
            },
        }

        recorder.record_sim_frame(
            timestamp=10.0,
            ships=ships,
            projectiles=[],
            torpedoes=[],
        )

        recording = recorder.get_recording()

        assert len(recording.sim_trace) == 1
        frame = recording.sim_trace[0]

        assert frame["t"] == 10.0
        assert "alpha" in frame["ships"]
        assert "beta" in frame["ships"]

        # Check alpha ship state
        alpha_state = frame["ships"]["alpha"]
        assert alpha_state["pos"] == [0.0, 0.0, 0.0]
        assert alpha_state["vel"] == [100.0, 0.0, 0.0]
        assert alpha_state["fwd"] == [1.0, 0.0, 0.0]
        assert alpha_state["thrust"] == 0.5
        assert alpha_state["maneuver"] == "INTERCEPT"
        assert alpha_state["destroyed"] is False
        assert alpha_state["hull"] == 95.0

    def test_sim_trace_destroyed_ships_remain(self, mock_battle_config, mock_captain_config):
        """Test that destroyed ships remain in sim_trace with destroyed=True and hull=0."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        # Frame 1: Both ships alive
        ships_frame1 = {
            "alpha": {
                "position": (0.0, 0.0, 0.0),
                "velocity": (100.0, 0.0, 0.0),
                "forward": (1.0, 0.0, 0.0),
                "is_destroyed": False,
                "hull_pct": 100.0,
            },
            "beta": {
                "position": (500000.0, 0.0, 0.0),
                "velocity": (-100.0, 0.0, 0.0),
                "forward": (-1.0, 0.0, 0.0),
                "is_destroyed": False,
                "hull_pct": 100.0,
            },
        }

        recorder.record_sim_frame(1.0, ships_frame1, [], [])

        # Frame 2: Beta ship destroyed
        ships_frame2 = {
            "alpha": {
                "position": (100.0, 0.0, 0.0),
                "velocity": (100.0, 0.0, 0.0),
                "forward": (1.0, 0.0, 0.0),
                "is_destroyed": False,
                "hull_pct": 90.0,
            },
            "beta": {
                "position": (499900.0, 0.0, 0.0),
                "velocity": (-100.0, 0.0, 0.0),
                "forward": (-1.0, 0.0, 0.0),
                "is_destroyed": True,
                "hull_pct": 0.0,
            },
        }

        recorder.record_sim_frame(2.0, ships_frame2, [], [])

        recording = recorder.get_recording()

        # Verify both frames exist
        assert len(recording.sim_trace) == 2

        # Frame 1: Both alive
        frame1 = recording.sim_trace[0]
        assert frame1["ships"]["alpha"]["destroyed"] is False
        assert frame1["ships"]["alpha"]["hull"] == 100.0
        assert frame1["ships"]["beta"]["destroyed"] is False
        assert frame1["ships"]["beta"]["hull"] == 100.0

        # Frame 2: Beta destroyed but still present
        frame2 = recording.sim_trace[1]
        assert "beta" in frame2["ships"]  # Destroyed ship still in trace
        assert frame2["ships"]["beta"]["destroyed"] is True
        assert frame2["ships"]["beta"]["hull"] == 0.0
        assert frame2["ships"]["alpha"]["destroyed"] is False

    def test_sim_trace_fleet_all_ships_tracked(self, mock_battle_config, mock_captain_config):
        """Test that all ships in a fleet battle are tracked in sim_trace."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        # Multi-ship fleet frame
        ships = {
            "alpha_1": {
                "position": (0.0, 0.0, 0.0),
                "velocity": (100.0, 0.0, 0.0),
                "forward": (1.0, 0.0, 0.0),
                "is_destroyed": False,
                "hull_pct": 100.0,
            },
            "alpha_2": {
                "position": (0.0, 1000.0, 0.0),
                "velocity": (100.0, 0.0, 0.0),
                "forward": (1.0, 0.0, 0.0),
                "is_destroyed": False,
                "hull_pct": 100.0,
            },
            "beta_1": {
                "position": (500000.0, 0.0, 0.0),
                "velocity": (-100.0, 0.0, 0.0),
                "forward": (-1.0, 0.0, 0.0),
                "is_destroyed": False,
                "hull_pct": 100.0,
            },
            "beta_2": {
                "position": (500000.0, 1000.0, 0.0),
                "velocity": (-100.0, 0.0, 0.0),
                "forward": (-1.0, 0.0, 0.0),
                "is_destroyed": True,  # Already destroyed
                "hull_pct": 0.0,
            },
        }

        recorder.record_sim_frame(10.0, ships, [], [])

        recording = recorder.get_recording()
        frame = recording.sim_trace[0]

        # Verify all four ships are in the trace
        assert len(frame["ships"]) == 4
        assert "alpha_1" in frame["ships"]
        assert "alpha_2" in frame["ships"]
        assert "beta_1" in frame["ships"]
        assert "beta_2" in frame["ships"]

        # Verify destroyed ship has correct status
        assert frame["ships"]["beta_2"]["destroyed"] is True
        assert frame["ships"]["beta_2"]["hull"] == 0.0

    def test_sim_trace_projectiles_and_torpedoes(self, mock_battle_config, mock_captain_config):
        """Test recording projectiles and torpedoes in sim_trace."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        ships = {
            "alpha": {
                "position": (0.0, 0.0, 0.0),
                "velocity": (0.0, 0.0, 0.0),
                "forward": (1.0, 0.0, 0.0),
                "is_destroyed": False,
                "hull_pct": 100.0,
            },
        }

        projectiles = [
            {
                "id": "proj_001",
                "position": (100000.0, 0.0, 0.0),
                "velocity": (8000.0, 0.0, 0.0),
                "mass_kg": 50.0,
                "source_ship_id": "alpha",
                "target_ship_id": "beta",
                "pd_engaged": True,
                "pd_ablation_kg": 2.5,
            },
        ]

        torpedoes = [
            {
                "id": "torp_001",
                "position": (200000.0, 0.0, 0.0),
                "velocity": (5000.0, 0.0, 0.0),
                "source_ship_id": "alpha",
                "target_ship_id": "beta",
                "dv_remaining_kps": 15.5,
                "heat_absorbed_j": 50000.0,
                "is_disabled": False,
            },
        ]

        recorder.record_sim_frame(20.0, ships, projectiles, torpedoes)

        recording = recorder.get_recording()
        frame = recording.sim_trace[0]

        # Verify projectile recorded
        assert len(frame["projectiles"]) == 1
        proj = frame["projectiles"][0]
        assert proj["id"] == "proj_001"
        assert proj["source"] == "alpha"
        assert proj["target"] == "beta"
        assert proj["pd_engaged"] is True
        assert proj["pd_damage_kg"] == 2.5

        # Verify torpedo recorded
        assert len(frame["torpedoes"]) == 1
        torp = frame["torpedoes"][0]
        assert torp["id"] == "torp_001"
        assert torp["source"] == "alpha"
        assert torp["dv_remaining"] == 15.5
        assert torp["pd_heat_j"] == 50000.0
        assert torp["disabled"] is False

    def test_sim_trace_not_recorded_when_not_recording(self, mock_battle_config, mock_captain_config):
        """Test that sim frames are not recorded when recording is not active."""
        recorder = BattleRecorder()

        # Don't start recording
        ships = {
            "alpha": {
                "position": (0.0, 0.0, 0.0),
                "velocity": (0.0, 0.0, 0.0),
                "forward": (1.0, 0.0, 0.0),
                "is_destroyed": False,
                "hull_pct": 100.0,
            },
        }

        recorder.record_sim_frame(10.0, ships, [], [])

        recording = recorder.get_recording()
        assert len(recording.sim_trace) == 0


# =============================================================================
# Test Module Damage Events
# =============================================================================

class TestModuleDamageEvents:
    """Tests for module damage and destruction events."""

    def test_record_module_damaged(self, mock_battle_config, mock_captain_config):
        """Test recording module damage event."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_module_damaged(
            timestamp=100.0,
            ship_id="beta",
            module_name="engine_1",
            damage_gj=0.75,
            destroyed=False,
        )

        events = [e for e in recorder.events if e.event_type == EventType.MODULE_DAMAGED]
        assert len(events) == 1

        event = events[0]
        assert event.timestamp == 100.0
        assert event.ship_id == "beta"
        assert event.data["module_name"] == "engine_1"
        assert event.data["damage_gj"] == 0.75
        assert event.data["destroyed"] is False

    def test_record_module_damaged_destroyed(self, mock_battle_config, mock_captain_config):
        """Test recording module damage event when module is destroyed."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_module_damaged(
            timestamp=105.0,
            ship_id="alpha",
            module_name="radiator_dorsal",
            damage_gj=2.0,
            destroyed=True,
        )

        events = [e for e in recorder.events if e.event_type == EventType.MODULE_DAMAGED]
        assert len(events) == 1

        event = events[0]
        assert event.data["destroyed"] is True

    def test_record_module_destroyed(self, mock_battle_config, mock_captain_config):
        """Test recording module destroyed event."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_module_destroyed(
            timestamp=110.0,
            ship_id="beta",
            module_name="weapon_turret_1",
        )

        events = [e for e in recorder.events if e.event_type == EventType.MODULE_DESTROYED]
        assert len(events) == 1

        event = events[0]
        assert event.timestamp == 110.0
        assert event.ship_id == "beta"
        assert event.data["module_name"] == "weapon_turret_1"

    def test_multiple_module_damage_events(self, mock_battle_config, mock_captain_config):
        """Test recording multiple module damage events across different ships."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        # Damage to various modules on different ships
        modules = [
            ("alpha", "engine_1", 0.5, False),
            ("alpha", "radiator_port", 1.2, True),
            ("beta", "weapon_0", 0.8, False),
            ("beta", "heatsink", 0.3, False),
            ("beta", "reactor", 2.5, True),
        ]

        for i, (ship_id, module, damage, destroyed) in enumerate(modules):
            recorder.record_module_damaged(
                timestamp=100.0 + i * 5,
                ship_id=ship_id,
                module_name=module,
                damage_gj=damage,
                destroyed=destroyed,
            )

        events = [e for e in recorder.events if e.event_type == EventType.MODULE_DAMAGED]
        assert len(events) == 5

        # Verify destroyed counts
        destroyed_events = [e for e in events if e.data["destroyed"]]
        assert len(destroyed_events) == 2


# =============================================================================
# Test Hit/Miss Events
# =============================================================================

class TestHitMissEvents:
    """Tests for hit and miss events."""

    def test_record_hit_complete_data(self, mock_battle_config, mock_captain_config):
        """Test recording a hit with all data fields."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_hit(
            timestamp=50.0,
            shooter_id="alpha",
            target_id="beta",
            weapon_slot="weapon_0",
            hit_location="nose",
            impact_angle_deg=15.0,
            kinetic_energy_gj=2.5,
            armor_ablation_cm=0.8,
            armor_remaining_cm=149.2,
            damage_to_hull_gj=0.5,
            penetrated=True,
            critical_hit=True,
        )

        events = [e for e in recorder.events if e.event_type == EventType.HIT]
        assert len(events) == 1

        event = events[0]
        assert event.timestamp == 50.0
        assert event.ship_id == "beta"  # Target ship is ship_id for hits
        assert event.data["shooter_id"] == "alpha"
        assert event.data["weapon_slot"] == "weapon_0"
        assert event.data["hit_location"] == "nose"
        assert event.data["impact_angle_deg"] == 15.0
        assert event.data["kinetic_energy_gj"] == 2.5
        assert event.data["armor_ablation_cm"] == 0.8
        assert event.data["armor_remaining_cm"] == 149.2
        assert event.data["damage_to_hull_gj"] == 0.5
        assert event.data["penetrated"] is True
        assert event.data["critical_hit"] is True

    def test_record_miss_complete_data(self, mock_battle_config, mock_captain_config):
        """Test recording a miss with all data fields."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_miss(
            timestamp=55.0,
            shooter_id="alpha",
            target_id="beta",
            weapon_slot="weapon_1",
            hit_probability=0.25,
            distance_km=400.0,
        )

        events = [e for e in recorder.events if e.event_type == EventType.MISS]
        assert len(events) == 1

        event = events[0]
        assert event.timestamp == 55.0
        assert event.ship_id == "alpha"  # Shooter is ship_id for misses
        assert event.data["target_id"] == "beta"
        assert event.data["weapon_slot"] == "weapon_1"
        assert event.data["hit_probability"] == 0.25
        assert event.data["distance_km"] == 400.0

    def test_record_shot_fired(self, mock_battle_config, mock_captain_config):
        """Test recording a shot fired event."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_shot_fired(
            timestamp=45.0,
            shooter_id="beta",
            target_id="alpha",
            weapon_slot="weapon_0",
            hit_probability=0.65,
            distance_km=350.0,
            projectile_energy_gj=2.5,
            muzzle_velocity_kps=8.0,
        )

        events = [e for e in recorder.events if e.event_type == EventType.SHOT_FIRED]
        assert len(events) == 1

        event = events[0]
        assert event.timestamp == 45.0
        assert event.ship_id == "beta"
        assert event.data["target_id"] == "alpha"
        assert event.data["weapon_slot"] == "weapon_0"
        assert event.data["hit_probability"] == 0.65
        assert event.data["distance_km"] == 350.0
        assert event.data["projectile_energy_gj"] == 2.5
        assert event.data["muzzle_velocity_kps"] == 8.0

    def test_hit_without_critical(self, mock_battle_config, mock_captain_config):
        """Test recording a hit without critical hit flag (default False)."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_hit(
            timestamp=60.0,
            shooter_id="alpha",
            target_id="beta",
            weapon_slot="weapon_0",
            hit_location="lateral",
            impact_angle_deg=45.0,
            kinetic_energy_gj=2.0,
            armor_ablation_cm=0.5,
            armor_remaining_cm=24.5,
            damage_to_hull_gj=0.0,
            penetrated=False,
        )

        events = [e for e in recorder.events if e.event_type == EventType.HIT]
        event = events[0]
        assert event.data["critical_hit"] is False


# =============================================================================
# Test Conversation/Message Events
# =============================================================================

class TestMessageEvents:
    """Tests for conversation and message events."""

    def test_record_message(self, mock_battle_config, mock_captain_config):
        """Test recording a message between captains."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_message(
            timestamp=30.0,
            sender_id="alpha",
            sender_name="Captain Alpha",
            ship_name="TIS Avenger",
            message="Prepare to be boarded!",
        )

        events = [e for e in recorder.events if e.event_type == EventType.MESSAGE]
        assert len(events) == 1

        event = events[0]
        assert event.timestamp == 30.0
        assert event.ship_id == "alpha"
        assert event.data["sender_name"] == "Captain Alpha"
        assert event.data["ship_name"] == "TIS Avenger"
        assert event.data["message"] == "Prepare to be boarded!"

    def test_record_multiple_messages(self, mock_battle_config, mock_captain_config):
        """Test recording multiple messages in a conversation."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        messages = [
            (30.0, "alpha", "Captain Alpha", "TIS Avenger", "This is your last chance to surrender!"),
            (60.0, "beta", "Captain Beta", "HFS Nemesis", "Never! We fight to the last!"),
            (90.0, "alpha", "Captain Alpha", "TIS Avenger", "Then prepare to meet your maker."),
        ]

        for ts, sender_id, sender_name, ship_name, msg in messages:
            recorder.record_message(ts, sender_id, sender_name, ship_name, msg)

        events = [e for e in recorder.events if e.event_type == EventType.MESSAGE]
        assert len(events) == 3

        # Verify chronological order
        assert events[0].timestamp == 30.0
        assert events[1].timestamp == 60.0
        assert events[2].timestamp == 90.0

        # Verify alternating senders
        assert events[0].ship_id == "alpha"
        assert events[1].ship_id == "beta"
        assert events[2].ship_id == "alpha"

    def test_record_surrender(self, mock_battle_config, mock_captain_config):
        """Test recording a surrender event."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_surrender(
            timestamp=300.0,
            ship_id="beta",
            captain_name="Captain Beta",
        )

        events = [e for e in recorder.events if e.event_type == EventType.SURRENDER]
        assert len(events) == 1

        event = events[0]
        assert event.timestamp == 300.0
        assert event.ship_id == "beta"
        assert event.data["captain_name"] == "Captain Beta"

    def test_record_draw_proposal(self, mock_battle_config, mock_captain_config):
        """Test recording a draw proposal event."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_draw_proposal(
            timestamp=450.0,
            ship_id="alpha",
            captain_name="Captain Alpha",
        )

        events = [e for e in recorder.events if e.event_type == EventType.DRAW_PROPOSAL]
        assert len(events) == 1

        event = events[0]
        assert event.timestamp == 450.0
        assert event.ship_id == "alpha"
        assert event.data["captain_name"] == "Captain Alpha"


# =============================================================================
# Test JSON Serialization
# =============================================================================

class TestJSONSerialization:
    """Tests for JSON serialization of recordings."""

    def test_recording_to_json_valid(self, mock_battle_config, mock_captain_config, mock_battle_result):
        """Test that recording produces valid JSON."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(name="Alpha"),
            mock_captain_config(name="Beta"),
        )

        recorder.record_message(30.0, "alpha", "Alpha", "Ship A", "Hello!")
        recorder.record_shot_fired(40.0, "alpha", "beta", "weapon_0", 0.5, 300.0, 2.0, 8.0)
        recorder.end_recording(mock_battle_result, 100.0)

        json_str = recorder.get_recording().to_json()

        # Verify valid JSON
        parsed = json.loads(json_str)

        assert parsed["alpha_name"] == "Alpha"
        assert parsed["beta_name"] == "Beta"
        assert len(parsed["events"]) > 0

    def test_save_recording_to_file(self, mock_battle_config, mock_captain_config, mock_battle_result):
        """Test saving recording to a file."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )
        recorder.end_recording(mock_battle_result, 100.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_battle.json"
            saved_path = recorder.save(str(filepath))

            assert Path(saved_path).exists()

            # Verify content
            with open(saved_path) as f:
                data = json.load(f)

            assert data["recording_version"] == "2.1"
            assert data["winner"] == "alpha"

    def test_save_recording_creates_directories(self, mock_battle_config, mock_captain_config, mock_battle_result):
        """Test that save creates parent directories if needed."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )
        recorder.end_recording(mock_battle_result, 100.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "nested" / "battle.json"
            saved_path = recorder.save(str(filepath))

            assert Path(saved_path).exists()

    def test_json_serialization_with_sim_trace(self, mock_battle_config, mock_captain_config, mock_battle_result):
        """Test JSON serialization includes sim_trace data."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        # Add some sim trace frames
        for i in range(3):
            ships = {
                "alpha": {
                    "position": (i * 100.0, 0.0, 0.0),
                    "velocity": (100.0, 0.0, 0.0),
                    "forward": (1.0, 0.0, 0.0),
                    "is_destroyed": False,
                    "hull_pct": 100.0 - i * 5,
                },
            }
            recorder.record_sim_frame(float(i), ships, [], [])

        recorder.end_recording(mock_battle_result, 100.0)

        json_str = recorder.get_recording().to_json()
        parsed = json.loads(json_str)

        assert len(parsed["sim_trace"]) == 3
        assert parsed["sim_trace"][0]["t"] == 0.0
        assert parsed["sim_trace"][1]["t"] == 1.0
        assert parsed["sim_trace"][2]["t"] == 2.0

    def test_json_preserves_event_order(self, mock_battle_config, mock_captain_config, mock_battle_result):
        """Test that JSON serialization preserves event order."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_maneuver(10.0, "alpha", "INTERCEPT", 1.0)
        recorder.record_shot_fired(20.0, "alpha", "beta", "weapon_0", 0.5, 300.0, 2.0, 8.0)
        recorder.record_hit(30.0, "alpha", "beta", "weapon_0", "nose", 10.0, 2.0, 0.5, 149.5, 0.2, False)
        recorder.end_recording(mock_battle_result, 100.0)

        json_str = recorder.get_recording().to_json()
        parsed = json.loads(json_str)

        events = parsed["events"]

        # Verify chronological order
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_battle_filename_basic(self):
        """Test creating a basic battle filename."""
        timestamp = datetime(2025, 3, 15, 14, 30, 45)
        filename = create_battle_filename(
            "model-alpha/v1",
            "model-beta/v2",
            timestamp,
        )

        assert filename.startswith("battle_")
        assert "v1" in filename
        assert "v2" in filename
        assert "20250315_143045" in filename
        assert filename.endswith(".json")

    def test_create_battle_filename_cleans_names(self):
        """Test that model names are cleaned in filename."""
        filename = create_battle_filename(
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o-mini",
        )

        # Should not contain slashes
        assert "/" not in filename
        # Dashes and dots should be replaced with underscores
        assert "claude_3_5_sonnet" in filename or "claude_3_5" in filename

    def test_create_battle_filename_truncates_long_names(self):
        """Test that very long model names are truncated."""
        filename = create_battle_filename(
            "very-long-model-name-that-exceeds-limit",
            "another-extremely-long-model-name",
        )

        # Model names should be truncated to 20 chars each
        # The filename should still be reasonable length
        assert len(filename) < 100

    def test_create_battle_filename_default_timestamp(self):
        """Test that filename uses current time when not specified."""
        filename = create_battle_filename("model-a", "model-b")

        # Should have today's date in some form
        today = datetime.now().strftime("%Y%m%d")
        assert today in filename


# =============================================================================
# Test Ship State Recording
# =============================================================================

class TestShipStateRecording:
    """Tests for detailed ship state recording."""

    def test_record_ship_state(self, mock_battle_config, mock_captain_config):
        """Test recording detailed ship state."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_ship_state(
            timestamp=50.0,
            ship_id="alpha",
            position=(1000.0, 500.0, 0.0),
            velocity=(100.0, 50.0, 0.0),
            forward=(0.9, 0.1, 0.0),
            hull_integrity=85.5,
            heat_percent=45.0,
            armor_nose_cm=145.0,
            armor_lateral_cm=23.5,
            armor_tail_cm=28.0,
            delta_v_remaining=450.0,
            radiators_extended=True,
        )

        events = [e for e in recorder.events if e.event_type == EventType.SHIP_STATE]
        assert len(events) == 1

        event = events[0]
        assert event.timestamp == 50.0
        assert event.ship_id == "alpha"
        assert event.data["position"] == (1000.0, 500.0, 0.0)
        assert event.data["velocity"] == (100.0, 50.0, 0.0)
        assert event.data["forward"] == (0.9, 0.1, 0.0)
        assert event.data["hull_integrity"] == 85.5
        assert event.data["heat_percent"] == 45.0
        assert event.data["armor"]["nose_cm"] == 145.0
        assert event.data["armor"]["lateral_cm"] == 23.5
        assert event.data["armor"]["tail_cm"] == 28.0
        assert event.data["delta_v_remaining"] == 450.0
        assert event.data["radiators_extended"] is True


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_events_not_recorded_when_not_recording(self, mock_battle_config, mock_captain_config):
        """Test that events are not recorded when recording is not started."""
        recorder = BattleRecorder()

        # Try to record without starting
        recorder.record_message(10.0, "alpha", "Captain", "Ship", "Test")
        recorder.record_shot_fired(20.0, "alpha", "beta", "weapon_0", 0.5, 300.0, 2.0, 8.0)

        assert len(recorder.events) == 0

    def test_events_not_recorded_after_end(self, mock_battle_config, mock_captain_config, mock_battle_result):
        """Test that events are not recorded after recording ends."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )
        recorder.end_recording(mock_battle_result, 100.0)

        initial_count = len(recorder.events)

        # Try to record after ending
        recorder.record_message(150.0, "alpha", "Captain", "Ship", "Test")

        assert len(recorder.events) == initial_count

    def test_restart_recording(self, mock_battle_config, mock_captain_config):
        """Test starting a new recording clears previous data."""
        recorder = BattleRecorder()

        # First recording
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(name="First Alpha"),
            mock_captain_config(name="First Beta"),
        )
        recorder.record_message(10.0, "alpha", "Captain", "Ship", "First message")

        # Start new recording (simulating a new battle)
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(name="Second Alpha"),
            mock_captain_config(name="Second Beta"),
        )

        recording = recorder.get_recording()

        assert recording.alpha_name == "Second Alpha"
        assert recording.beta_name == "Second Beta"
        # Events should be reset (only BATTLE_START from new recording)
        assert len(recorder.events) == 1
        assert recorder.events[0].event_type == EventType.BATTLE_START

    def test_empty_sim_trace(self, mock_battle_config, mock_captain_config, mock_battle_result):
        """Test recording with no sim trace frames."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )
        recorder.end_recording(mock_battle_result, 100.0)

        recording = recorder.get_recording()
        assert recording.sim_trace == []

        # Should still produce valid JSON
        json_str = recording.to_json()
        parsed = json.loads(json_str)
        assert parsed["sim_trace"] == []

    def test_recording_with_unicode_messages(self, mock_battle_config, mock_captain_config):
        """Test recording messages with unicode characters."""
        recorder = BattleRecorder()
        recorder.start_recording(
            mock_battle_config,
            mock_captain_config(),
            mock_captain_config(),
        )

        recorder.record_message(
            10.0,
            "alpha",
            "Kapitan",
            "KMS Scharnhorst",
            "Feuer frei!"
        )

        events = [e for e in recorder.events if e.event_type == EventType.MESSAGE]
        assert events[0].data["message"] == "Feuer frei!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
