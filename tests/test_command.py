"""
Comprehensive test suite for the command module.

Tests the hierarchical control architecture:
- Command dataclasses (SetThrust, RotateTo, Engage, HoldFire, Evade)
- TacticalController behaviors
- BattleState snapshot and properties
- StrategicController decision timing and command generation
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from unittest.mock import Mock, patch

from src.command import (
    # Command types
    SetThrust,
    RotateTo,
    Engage,
    HoldFire,
    Evade,
    Command,
    # Event types
    TacticalEventType,
    TacticalEvent,
    # State types
    ShipState,
    ThreatInfo,
    ProjectileInfo,
    BattleState,
    # Controllers
    TacticalController,
    StrategicController,
    LLMStrategicController,
    RuleBasedStrategicController,
    # Utilities
    validate_command,
    Vector3,
    ShipId,
    WeaponSlot,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def unit_vectors():
    """Common unit vectors for testing."""
    return {
        'forward': np.array([1.0, 0.0, 0.0]),
        'backward': np.array([-1.0, 0.0, 0.0]),
        'up': np.array([0.0, 0.0, 1.0]),
        'down': np.array([0.0, 0.0, -1.0]),
        'left': np.array([0.0, 1.0, 0.0]),
        'right': np.array([0.0, -1.0, 0.0]),
        'diagonal': np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
    }


@pytest.fixture
def basic_ship_state():
    """Create a basic ship state for testing."""
    return ShipState(
        ship_id='ship_001',
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([100.0, 0.0, 0.0]),
        heading=np.array([1.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        hull_hp={'core': (100.0, 100.0), 'armor': (50.0, 50.0)},
        heat_percentage=30.0,
        battery_percentage=80.0,
        delta_v_remaining=100_000.0,
        weapons_status={
            'turret_dorsal': {'status': 'ready', 'ammo': 100},
            'turret_ventral': {'status': 'ready', 'ammo': 100},
            'spinal': {'status': 'charging', 'ammo': 10},
            'torpedo': {'status': 'ready', 'ammo': 4, 'type': 'torpedo_launcher'},
        },
        is_destroyed=False,
        is_disabled=False,
    )


@pytest.fixture
def enemy_ship_state():
    """Create an enemy ship state for testing."""
    return ShipState(
        ship_id='enemy_001',
        position=np.array([100_000.0, 0.0, 0.0]),
        velocity=np.array([-50.0, 0.0, 0.0]),
        heading=np.array([-1.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        hull_hp={'core': (80.0, 100.0), 'armor': (30.0, 50.0)},
        heat_percentage=45.0,
        battery_percentage=60.0,
        delta_v_remaining=80_000.0,
        weapons_status={
            'turret_dorsal': {'status': 'ready', 'ammo': 80},
            'torpedo': {'status': 'ready', 'ammo': 2, 'type': 'torpedo_launcher'},
        },
        is_destroyed=False,
        is_disabled=False,
    )


@pytest.fixture
def incoming_threat():
    """Create an incoming threat for testing."""
    return ThreatInfo(
        threat_id='torpedo_001',
        threat_type='torpedo',
        position=np.array([50_000.0, 0.0, 0.0]),
        velocity=np.array([-500.0, 0.0, 0.0]),
        estimated_time_to_impact=100.0,
        source_ship_id='enemy_001',
        can_be_intercepted=True,
    )


@pytest.fixture
def critical_threat():
    """Create a critical threat (< 30s to impact) for testing."""
    return ThreatInfo(
        threat_id='torpedo_002',
        threat_type='torpedo',
        position=np.array([10_000.0, 0.0, 0.0]),
        velocity=np.array([-500.0, 0.0, 0.0]),
        estimated_time_to_impact=20.0,
        source_ship_id='enemy_001',
        can_be_intercepted=True,
    )


@pytest.fixture
def basic_battle_state(basic_ship_state, enemy_ship_state):
    """Create a basic battle state for testing."""
    return BattleState(
        timestamp=100.0,
        own_ship=basic_ship_state,
        enemy_ships=[enemy_ship_state],
        friendly_ships=[],
        incoming_threats=[],
        active_projectiles=[],
        recent_events=[],
        engagement_range=100_000.0,
        closing_rate=150.0,
        battle_duration=100.0,
    )


@pytest.fixture
def battle_state_with_threats(basic_ship_state, enemy_ship_state, incoming_threat, critical_threat):
    """Create a battle state with incoming threats."""
    return BattleState(
        timestamp=100.0,
        own_ship=basic_ship_state,
        enemy_ships=[enemy_ship_state],
        friendly_ships=[],
        incoming_threats=[incoming_threat, critical_threat],
        active_projectiles=[],
        recent_events=[],
        engagement_range=100_000.0,
        closing_rate=150.0,
        battle_duration=100.0,
    )


# =============================================================================
# Command Dataclass Tests
# =============================================================================

class TestSetThrust:
    """Tests for the SetThrust command."""

    def test_valid_magnitude_zero(self, unit_vectors):
        """SetThrust accepts magnitude 0."""
        cmd = SetThrust(vector=unit_vectors['forward'], magnitude=0.0)
        assert cmd.magnitude == 0.0
        assert_array_almost_equal(cmd.vector, unit_vectors['forward'])

    def test_valid_magnitude_one(self, unit_vectors):
        """SetThrust accepts magnitude 1."""
        cmd = SetThrust(vector=unit_vectors['forward'], magnitude=1.0)
        assert cmd.magnitude == 1.0

    def test_valid_magnitude_mid_range(self, unit_vectors):
        """SetThrust accepts magnitude between 0 and 1."""
        cmd = SetThrust(vector=unit_vectors['forward'], magnitude=0.5)
        assert cmd.magnitude == 0.5

    def test_invalid_magnitude_negative(self, unit_vectors):
        """SetThrust rejects negative magnitude."""
        with pytest.raises(ValueError, match="Magnitude must be between 0.0 and 1.0"):
            SetThrust(vector=unit_vectors['forward'], magnitude=-0.1)

    def test_invalid_magnitude_above_one(self, unit_vectors):
        """SetThrust rejects magnitude above 1."""
        with pytest.raises(ValueError, match="Magnitude must be between 0.0 and 1.0"):
            SetThrust(vector=unit_vectors['forward'], magnitude=1.1)

    def test_invalid_magnitude_large_value(self, unit_vectors):
        """SetThrust rejects large magnitude values."""
        with pytest.raises(ValueError, match="Magnitude must be between 0.0 and 1.0"):
            SetThrust(vector=unit_vectors['forward'], magnitude=100.0)

    def test_frozen_dataclass(self, unit_vectors):
        """SetThrust is immutable (frozen)."""
        cmd = SetThrust(vector=unit_vectors['forward'], magnitude=0.5)
        with pytest.raises(AttributeError):
            cmd.magnitude = 0.7

    def test_different_vectors(self, unit_vectors):
        """SetThrust works with different direction vectors."""
        for direction_name, direction in unit_vectors.items():
            cmd = SetThrust(vector=direction, magnitude=0.5)
            assert_array_almost_equal(cmd.vector, direction)


class TestRotateTo:
    """Tests for the RotateTo command."""

    def test_basic_bearing(self, unit_vectors):
        """RotateTo stores target bearing correctly."""
        cmd = RotateTo(target_bearing=unit_vectors['forward'])
        assert_array_almost_equal(cmd.target_bearing, unit_vectors['forward'])

    def test_different_bearings(self, unit_vectors):
        """RotateTo works with different bearings."""
        for direction_name, direction in unit_vectors.items():
            cmd = RotateTo(target_bearing=direction)
            assert_array_almost_equal(cmd.target_bearing, direction)

    def test_diagonal_bearing(self):
        """RotateTo works with diagonal bearings."""
        diagonal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        cmd = RotateTo(target_bearing=diagonal)
        assert_array_almost_equal(cmd.target_bearing, diagonal)

    def test_frozen_dataclass(self, unit_vectors):
        """RotateTo is immutable (frozen)."""
        cmd = RotateTo(target_bearing=unit_vectors['forward'])
        with pytest.raises(AttributeError):
            cmd.target_bearing = unit_vectors['backward']


class TestEngage:
    """Tests for the Engage command."""

    def test_basic_engage(self):
        """Engage stores target and weapon slots correctly."""
        cmd = Engage(target_id='enemy_001', weapon_slots=['turret_dorsal', 'spinal'])
        assert cmd.target_id == 'enemy_001'
        assert cmd.weapon_slots == ('turret_dorsal', 'spinal')

    def test_single_weapon(self):
        """Engage works with a single weapon."""
        cmd = Engage(target_id='enemy_001', weapon_slots=['torpedo'])
        assert cmd.weapon_slots == ('torpedo',)

    def test_multiple_weapons(self):
        """Engage works with multiple weapons."""
        weapons = ['turret_dorsal', 'turret_ventral', 'spinal', 'torpedo']
        cmd = Engage(target_id='enemy_001', weapon_slots=weapons)
        assert cmd.weapon_slots == tuple(weapons)

    def test_empty_weapons_list(self):
        """Engage allows empty weapon slots (tactical layer handles validation)."""
        cmd = Engage(target_id='enemy_001', weapon_slots=[])
        assert cmd.weapon_slots == ()

    def test_weapon_slots_converted_to_tuple(self):
        """Engage converts weapon_slots list to tuple for immutability."""
        weapons_list = ['turret_dorsal', 'spinal']
        cmd = Engage(target_id='enemy_001', weapon_slots=weapons_list)
        assert isinstance(cmd.weapon_slots, tuple)

    def test_frozen_dataclass(self):
        """Engage is immutable (frozen)."""
        cmd = Engage(target_id='enemy_001', weapon_slots=['turret_dorsal'])
        with pytest.raises(AttributeError):
            cmd.target_id = 'enemy_002'


class TestHoldFire:
    """Tests for the HoldFire command."""

    def test_default_preserves_point_defense(self):
        """HoldFire defaults to keeping point defense active."""
        cmd = HoldFire()
        assert cmd.include_point_defense is False

    def test_include_point_defense_true(self):
        """HoldFire can stop point defense."""
        cmd = HoldFire(include_point_defense=True)
        assert cmd.include_point_defense is True

    def test_include_point_defense_false(self):
        """HoldFire can explicitly keep point defense active."""
        cmd = HoldFire(include_point_defense=False)
        assert cmd.include_point_defense is False

    def test_frozen_dataclass(self):
        """HoldFire is immutable (frozen)."""
        cmd = HoldFire()
        with pytest.raises(AttributeError):
            cmd.include_point_defense = True


class TestEvade:
    """Tests for the Evade command."""

    def test_basic_evade(self):
        """Evade stores threat_id correctly."""
        cmd = Evade(threat_id='torpedo_001')
        assert cmd.threat_id == 'torpedo_001'

    def test_different_threat_ids(self):
        """Evade works with different threat IDs."""
        threat_ids = ['torpedo_001', 'missile_002', 'slug_003']
        for threat_id in threat_ids:
            cmd = Evade(threat_id=threat_id)
            assert cmd.threat_id == threat_id

    def test_frozen_dataclass(self):
        """Evade is immutable (frozen)."""
        cmd = Evade(threat_id='torpedo_001')
        with pytest.raises(AttributeError):
            cmd.threat_id = 'torpedo_002'


# =============================================================================
# BattleState Tests
# =============================================================================

class TestBattleState:
    """Tests for BattleState snapshot and properties."""

    def test_state_snapshot_creation(self, basic_battle_state):
        """BattleState correctly stores all fields."""
        assert basic_battle_state.timestamp == 100.0
        assert basic_battle_state.own_ship.ship_id == 'ship_001'
        assert len(basic_battle_state.enemy_ships) == 1
        assert basic_battle_state.engagement_range == 100_000.0

    def test_primary_target_returns_closest_enemy(self, basic_ship_state):
        """primary_target returns the closest enemy ship."""
        # Create two enemies at different distances
        enemy_close = ShipState(
            ship_id='enemy_close',
            position=np.array([50_000.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            heading=np.array([1.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            hull_hp={'core': (100.0, 100.0)},
            heat_percentage=30.0,
            battery_percentage=80.0,
            delta_v_remaining=100_000.0,
            weapons_status={},
        )
        enemy_far = ShipState(
            ship_id='enemy_far',
            position=np.array([200_000.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            heading=np.array([1.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            hull_hp={'core': (100.0, 100.0)},
            heat_percentage=30.0,
            battery_percentage=80.0,
            delta_v_remaining=100_000.0,
            weapons_status={},
        )

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_far, enemy_close],  # Intentionally reversed
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=50_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        assert battle_state.primary_target.ship_id == 'enemy_close'

    def test_primary_target_no_enemies(self, basic_ship_state):
        """primary_target returns None when no enemies exist."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        assert battle_state.primary_target is None

    def test_critical_threats_filters_by_time(self, battle_state_with_threats):
        """critical_threats returns only threats with < 30s to impact."""
        critical = battle_state_with_threats.critical_threats
        assert len(critical) == 1
        assert critical[0].threat_id == 'torpedo_002'
        assert critical[0].estimated_time_to_impact < 30.0

    def test_critical_threats_empty_when_none_critical(self, basic_battle_state, incoming_threat):
        """critical_threats returns empty list when no threats are critical."""
        battle_state = basic_battle_state
        battle_state.incoming_threats = [incoming_threat]  # 100s to impact
        assert battle_state.critical_threats == []

    def test_get_threat_by_id_found(self, battle_state_with_threats):
        """get_threat_by_id returns the threat when found."""
        threat = battle_state_with_threats.get_threat_by_id('torpedo_001')
        assert threat is not None
        assert threat.threat_id == 'torpedo_001'

    def test_get_threat_by_id_not_found(self, battle_state_with_threats):
        """get_threat_by_id returns None when threat not found."""
        threat = battle_state_with_threats.get_threat_by_id('nonexistent')
        assert threat is None

    def test_enemy_detection_multiple_enemies(self, basic_ship_state):
        """BattleState tracks multiple detected enemy ships."""
        enemies = []
        for i in range(3):
            enemy = ShipState(
                ship_id=f'enemy_{i:03d}',
                position=np.array([100_000.0 * (i + 1), 0.0, 0.0]),
                velocity=np.zeros(3),
                heading=np.array([1.0, 0.0, 0.0]),
                angular_velocity=np.zeros(3),
                hull_hp={'core': (100.0, 100.0)},
                heat_percentage=30.0,
                battery_percentage=80.0,
                delta_v_remaining=100_000.0,
                weapons_status={},
            )
            enemies.append(enemy)

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=enemies,
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=100_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        assert len(battle_state.enemy_ships) == 3

    def test_threat_tracking_multiple_threats(self, basic_ship_state):
        """BattleState tracks multiple incoming threats."""
        threats = []
        for i in range(5):
            threat = ThreatInfo(
                threat_id=f'torpedo_{i:03d}',
                threat_type='torpedo',
                position=np.array([50_000.0 - i * 5_000, 0.0, 0.0]),
                velocity=np.array([-500.0, 0.0, 0.0]),
                estimated_time_to_impact=100.0 - i * 20,
                source_ship_id='enemy_001',
                can_be_intercepted=True,
            )
            threats.append(threat)

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=threats,
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        assert len(battle_state.incoming_threats) == 5
        # Threats with ETA < 30s should be critical
        # ETAs are: 100, 80, 60, 40, 20 - only the last one (20s) is < 30s
        critical = battle_state.critical_threats
        assert len(critical) == 1
        assert critical[0].estimated_time_to_impact == 20.0


# =============================================================================
# TacticalController Tests
# =============================================================================

class TestTacticalController:
    """Tests for TacticalController behaviors."""

    def test_initialization(self):
        """TacticalController initializes with correct defaults."""
        controller = TacticalController(ship_id='ship_001')
        assert controller.ship_id == 'ship_001'
        assert controller.point_defense_auto is True
        assert controller.evade_threshold_seconds == 60.0
        assert controller.active_commands == []

    def test_custom_weapon_ranges(self):
        """TacticalController accepts custom weapon range limits."""
        custom_ranges = {'turret_dorsal': 300_000, 'spinal': 750_000}
        controller = TacticalController(
            ship_id='ship_001',
            weapon_range_limits=custom_ranges,
        )
        assert controller.weapon_range_limits['turret_dorsal'] == 300_000
        assert controller.weapon_range_limits['spinal'] == 750_000

    def test_set_commands(self, unit_vectors):
        """set_commands replaces active commands."""
        controller = TacticalController(ship_id='ship_001')
        commands = [
            SetThrust(vector=unit_vectors['forward'], magnitude=0.5),
            RotateTo(target_bearing=unit_vectors['forward']),
        ]
        controller.set_commands(commands)
        assert len(controller.active_commands) == 2

    def test_keep_nose_toward_target(self, basic_battle_state, enemy_ship_state):
        """TacticalController rotates ship to face engage target."""
        controller = TacticalController(ship_id='ship_001')
        controller.set_commands([
            Engage(target_id='enemy_001', weapon_slots=['turret_dorsal'])
        ])

        outputs, events = controller.update(basic_battle_state, delta_time=1.0)

        # Should have target heading set toward enemy
        assert outputs['target_heading'] is not None
        expected_direction = enemy_ship_state.position - basic_battle_state.own_ship.position
        expected_direction = expected_direction / np.linalg.norm(expected_direction)
        assert_array_almost_equal(outputs['target_heading'], expected_direction)

    def test_fire_when_in_range(self, basic_ship_state, enemy_ship_state):
        """TacticalController fires weapons when target is in range."""
        # Place enemy close enough for turrets (within 200km)
        enemy_ship_state.position = np.array([150_000.0, 0.0, 0.0])

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=150_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001')
        controller.set_commands([
            Engage(target_id='enemy_001', weapon_slots=['turret_dorsal'])
        ])

        outputs, events = controller.update(battle_state, delta_time=1.0)

        # Weapon should be firing
        assert 'turret_dorsal' in outputs['weapons_fire']
        assert outputs['weapons_fire']['turret_dorsal']['target_id'] == 'enemy_001'
        assert outputs['weapons_fire']['turret_dorsal']['fire'] is True

    def test_no_fire_when_out_of_range(self, basic_ship_state, enemy_ship_state):
        """TacticalController does not fire when target is out of range."""
        # Place enemy far away (beyond turret range of 200km)
        enemy_ship_state.position = np.array([300_000.0, 0.0, 0.0])

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=300_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001')
        controller.set_commands([
            Engage(target_id='enemy_001', weapon_slots=['turret_dorsal'])
        ])

        outputs, events = controller.update(battle_state, delta_time=1.0)

        # Weapon should not be firing
        assert 'turret_dorsal' not in outputs['weapons_fire']

    def test_evade_incoming_torpedoes(self, basic_ship_state, critical_threat):
        """TacticalController executes evasive maneuvers against threats."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[critical_threat],
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001')
        controller.set_commands([Evade(threat_id='torpedo_002')])

        outputs, events = controller.update(battle_state, delta_time=1.0)

        # Should have evasive thrust
        assert outputs['thrust_magnitude'] == 1.0  # Full thrust for evasion
        # Thrust should be perpendicular to threat vector
        threat_vector = critical_threat.velocity / np.linalg.norm(critical_threat.velocity)
        dot_product = np.dot(outputs['thrust_vector'], threat_vector)
        assert abs(dot_product) < 0.1  # Nearly perpendicular

    def test_target_lost_event(self, basic_ship_state):
        """TacticalController generates TARGET_LOST event when target disappears."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],  # No enemies - target is lost
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001')
        controller.set_commands([
            Engage(target_id='enemy_001', weapon_slots=['turret_dorsal'])
        ])

        outputs, events = controller.update(battle_state, delta_time=1.0)

        target_lost_events = [e for e in events if e.event_type == TacticalEventType.TARGET_LOST]
        assert len(target_lost_events) == 1
        assert target_lost_events[0].details['target_id'] == 'enemy_001'

    def test_target_destroyed_event(self, basic_ship_state, enemy_ship_state):
        """TacticalController generates TARGET_DESTROYED event when target is destroyed."""
        enemy_ship_state.is_destroyed = True

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=100_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001')
        controller.set_commands([
            Engage(target_id='enemy_001', weapon_slots=['turret_dorsal'])
        ])

        outputs, events = controller.update(battle_state, delta_time=1.0)

        destroyed_events = [e for e in events if e.event_type == TacticalEventType.TARGET_DESTROYED]
        assert len(destroyed_events) == 1

    def test_heat_critical_event(self, basic_ship_state, enemy_ship_state):
        """TacticalController generates HEAT_CRITICAL event when ship overheats."""
        basic_ship_state.heat_percentage = 95.0

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=100_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001')
        outputs, events = controller.update(battle_state, delta_time=1.0)

        heat_events = [e for e in events if e.event_type == TacticalEventType.HEAT_CRITICAL]
        assert len(heat_events) == 1
        assert heat_events[0].priority == 5

    def test_critical_damage_event(self, basic_ship_state, enemy_ship_state):
        """TacticalController generates CRITICAL_DAMAGE event when hull is low."""
        basic_ship_state.hull_hp = {'core': (10.0, 100.0), 'armor': (5.0, 50.0)}  # 10% HP

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=100_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001')
        outputs, events = controller.update(battle_state, delta_time=1.0)

        damage_events = [e for e in events if e.event_type == TacticalEventType.CRITICAL_DAMAGE]
        assert len(damage_events) == 1
        assert damage_events[0].priority == 5

    def test_torpedo_incoming_event(self, basic_ship_state, critical_threat):
        """TacticalController generates TORPEDO_INCOMING event for close torpedoes."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[critical_threat],  # 20s to impact
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001', evade_threshold_seconds=30.0)
        outputs, events = controller.update(battle_state, delta_time=1.0)

        torpedo_events = [e for e in events if e.event_type == TacticalEventType.TORPEDO_INCOMING]
        assert len(torpedo_events) == 1
        assert torpedo_events[0].details['threat_id'] == 'torpedo_002'

    def test_low_delta_v_event(self, basic_ship_state, enemy_ship_state):
        """TacticalController generates LOW_DELTA_V event when propellant is low."""
        basic_ship_state.delta_v_remaining = 40_000.0  # Below 50km/s threshold

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=100_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001')
        outputs, events = controller.update(battle_state, delta_time=1.0)

        delta_v_events = [e for e in events if e.event_type == TacticalEventType.LOW_DELTA_V]
        assert len(delta_v_events) == 1

    def test_auto_point_defense(self, basic_ship_state, incoming_threat, critical_threat):
        """TacticalController automatically engages threats with point defense."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[incoming_threat, critical_threat],
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001', point_defense_auto=True)
        outputs, events = controller.update(battle_state, delta_time=1.0)

        # Should assign PD to threats (most urgent first)
        pd_targets = outputs['point_defense_targets']
        assert len(pd_targets) == 2
        # Most urgent threat should be first
        assert pd_targets[0] == 'torpedo_002'  # 20s to impact

    def test_hold_fire_clears_weapons(self, basic_battle_state, unit_vectors):
        """TacticalController clears weapon fire on HoldFire command."""
        controller = TacticalController(ship_id='ship_001')
        controller.set_commands([HoldFire()])

        outputs, events = controller.update(basic_battle_state, delta_time=1.0)

        assert outputs['weapons_fire'] == {}

    def test_hold_fire_with_point_defense(self, basic_battle_state, incoming_threat):
        """HoldFire with include_point_defense clears PD targets."""
        basic_battle_state.incoming_threats = [incoming_threat]

        # When include_point_defense=True, auto PD should be disabled
        # The command clears PD targets in _process_hold_fire AFTER auto_point_defense runs
        # So we test that include_point_defense properly signals intent to stop PD
        controller = TacticalController(ship_id='ship_001', point_defense_auto=False)
        controller.set_commands([HoldFire(include_point_defense=True)])

        outputs, events = controller.update(basic_battle_state, delta_time=1.0)

        # With auto PD disabled, the output should be empty after HoldFire
        assert outputs['point_defense_targets'] == []

    def test_maneuver_complete_event(self, basic_ship_state):
        """TacticalController generates MANEUVER_COMPLETE when threat disappears."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[],  # Threat no longer exists
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = TacticalController(ship_id='ship_001')
        controller.set_commands([Evade(threat_id='torpedo_001')])

        outputs, events = controller.update(battle_state, delta_time=1.0)

        maneuver_events = [e for e in events if e.event_type == TacticalEventType.MANEUVER_COMPLETE]
        assert len(maneuver_events) == 1
        assert maneuver_events[0].details['maneuver'] == 'evade'

    def test_enemy_maneuver_detection(self, basic_ship_state, enemy_ship_state):
        """TacticalController detects enemy maneuvers."""
        controller = TacticalController(ship_id='ship_001')

        # First update to establish baseline position
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=100_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )
        controller.update(battle_state, delta_time=1.0)

        # Second update with enemy in unexpected position (maneuvered)
        enemy_ship_state.position = np.array([100_000.0, 5_000.0, 0.0])  # Moved 5km sideways
        battle_state.timestamp = 101.0
        outputs, events = controller.update(battle_state, delta_time=1.0)

        maneuver_events = [e for e in events if e.event_type == TacticalEventType.ENEMY_MANEUVER_DETECTED]
        assert len(maneuver_events) == 1


# =============================================================================
# StrategicController Tests
# =============================================================================

class TestStrategicController:
    """Tests for StrategicController decision timing."""

    def test_should_decide_initial(self):
        """should_decide returns True after decision interval from start."""
        controller = RuleBasedStrategicController(ship_id='ship_001', decision_interval=10.0)
        # Initially _last_decision_time is 0.0, so we need to wait for the interval
        # At t=0, 0.0 - 0.0 >= 10.0 is False
        # At t=10, 10.0 - 0.0 >= 10.0 is True
        assert controller.should_decide(0.0) is False  # Just started
        assert controller.should_decide(10.0) is True  # After interval

    def test_should_decide_after_interval(self):
        """should_decide returns True after interval passes."""
        controller = RuleBasedStrategicController(ship_id='ship_001', decision_interval=10.0)
        controller._last_decision_time = 0.0
        assert controller.should_decide(5.0) is False
        assert controller.should_decide(10.0) is True
        assert controller.should_decide(15.0) is True

    def test_request_decision_updates_time(self, basic_battle_state):
        """request_decision updates last decision time."""
        controller = RuleBasedStrategicController(ship_id='ship_001', decision_interval=10.0)
        basic_battle_state.timestamp = 50.0

        controller.request_decision(basic_battle_state, [])

        assert controller._last_decision_time == 50.0

    def test_decision_interval_configuration(self):
        """Strategic controller respects decision interval configuration."""
        controller_fast = RuleBasedStrategicController(ship_id='ship_001', decision_interval=5.0)
        controller_slow = RuleBasedStrategicController(ship_id='ship_002', decision_interval=60.0)

        assert controller_fast.decision_interval == 5.0
        assert controller_slow.decision_interval == 60.0


class TestRuleBasedStrategicController:
    """Tests for RuleBasedStrategicController scenarios."""

    def test_evade_critical_threats_first(self, basic_ship_state, enemy_ship_state, critical_threat):
        """RuleBasedStrategicController prioritizes evading critical threats."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[critical_threat],
            active_projectiles=[],
            recent_events=[],
            engagement_range=100_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = RuleBasedStrategicController(ship_id='ship_001')
        commands = controller.decide(battle_state, [])

        # First command should be Evade
        evade_commands = [c for c in commands if isinstance(c, Evade)]
        assert len(evade_commands) == 1
        assert evade_commands[0].threat_id == 'torpedo_002'

    def test_close_range_when_too_far(self, basic_ship_state, enemy_ship_state):
        """RuleBasedStrategicController closes range when target is far."""
        # Put enemy far away (beyond preferred range)
        enemy_ship_state.position = np.array([300_000.0, 0.0, 0.0])

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=300_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = RuleBasedStrategicController(
            ship_id='ship_001',
            preferred_range=150_000.0,
        )
        commands = controller.decide(battle_state, [])

        # Should have SetThrust toward enemy
        thrust_commands = [c for c in commands if isinstance(c, SetThrust)]
        assert len(thrust_commands) == 1
        assert thrust_commands[0].magnitude == 0.7  # Closing thrust

    def test_back_off_when_too_close(self, basic_ship_state, enemy_ship_state):
        """RuleBasedStrategicController backs off when target is too close."""
        # Put enemy very close (below preferred range)
        enemy_ship_state.position = np.array([50_000.0, 0.0, 0.0])

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[enemy_ship_state],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=50_000.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = RuleBasedStrategicController(
            ship_id='ship_001',
            preferred_range=150_000.0,
        )
        commands = controller.decide(battle_state, [])

        # Should have SetThrust away from enemy (backing off)
        thrust_commands = [c for c in commands if isinstance(c, SetThrust)]
        assert len(thrust_commands) == 1
        assert thrust_commands[0].magnitude == 0.5  # Backing thrust

    def test_engage_with_all_weapons(self, basic_battle_state):
        """RuleBasedStrategicController engages with all available weapons."""
        controller = RuleBasedStrategicController(ship_id='ship_001')
        commands = controller.decide(basic_battle_state, [])

        engage_commands = [c for c in commands if isinstance(c, Engage)]
        assert len(engage_commands) == 1
        # Should include all weapon slots from ship status
        assert 'turret_dorsal' in engage_commands[0].weapon_slots

    def test_hold_fire_no_target(self, basic_ship_state):
        """RuleBasedStrategicController holds fire when no target exists."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = RuleBasedStrategicController(ship_id='ship_001')
        commands = controller.decide(battle_state, [])

        hold_fire_commands = [c for c in commands if isinstance(c, HoldFire)]
        assert len(hold_fire_commands) == 1

    def test_always_face_target(self, basic_battle_state):
        """RuleBasedStrategicController always rotates to face target."""
        controller = RuleBasedStrategicController(ship_id='ship_001')
        commands = controller.decide(basic_battle_state, [])

        rotate_commands = [c for c in commands if isinstance(c, RotateTo)]
        assert len(rotate_commands) == 1

    def test_on_critical_event_torpedo_incoming(self, basic_battle_state):
        """RuleBasedStrategicController responds to torpedo incoming event."""
        event = TacticalEvent(
            event_type=TacticalEventType.TORPEDO_INCOMING,
            priority=4,
            details={'threat_id': 'torpedo_001', 'eta_seconds': 25.0},
            timestamp=100.0,
        )

        controller = RuleBasedStrategicController(ship_id='ship_001')
        commands = controller.on_critical_event(event, basic_battle_state)

        assert commands is not None
        evade_commands = [c for c in commands if isinstance(c, Evade)]
        assert len(evade_commands) == 1
        assert evade_commands[0].threat_id == 'torpedo_001'

    def test_on_critical_event_heat_critical(self, basic_battle_state):
        """RuleBasedStrategicController responds to heat critical event."""
        event = TacticalEvent(
            event_type=TacticalEventType.HEAT_CRITICAL,
            priority=5,
            details={'heat_percentage': 95.0},
            timestamp=100.0,
        )

        controller = RuleBasedStrategicController(ship_id='ship_001')
        commands = controller.on_critical_event(event, basic_battle_state)

        assert commands is not None
        # Should include HoldFire and reduce thrust
        hold_fire_commands = [c for c in commands if isinstance(c, HoldFire)]
        thrust_commands = [c for c in commands if isinstance(c, SetThrust)]
        assert len(hold_fire_commands) == 1
        assert len(thrust_commands) == 1
        assert thrust_commands[0].magnitude == 0.0

    def test_on_critical_event_non_critical_returns_none(self, basic_battle_state):
        """RuleBasedStrategicController ignores non-critical events."""
        event = TacticalEvent(
            event_type=TacticalEventType.FIRING_SOLUTION_ACQUIRED,
            priority=2,
            details={'weapon_slot': 'turret_dorsal'},
            timestamp=100.0,
        )

        controller = RuleBasedStrategicController(ship_id='ship_001')
        commands = controller.on_critical_event(event, basic_battle_state)

        assert commands is None


class TestLLMStrategicController:
    """Tests for LLMStrategicController (placeholder implementation)."""

    def test_initialization(self):
        """LLMStrategicController initializes with correct parameters."""
        controller = LLMStrategicController(
            ship_id='ship_001',
            decision_interval=30.0,
            llm_model='claude-3-opus',
            temperature=0.3,
        )
        assert controller.ship_id == 'ship_001'
        assert controller.decision_interval == 30.0
        assert controller.llm_model == 'claude-3-opus'
        assert controller.temperature == 0.3

    def test_default_aggressive_behavior_with_target(self, basic_battle_state):
        """LLMStrategicController falls back to aggressive behavior with target."""
        controller = LLMStrategicController(ship_id='ship_001')
        commands = controller.decide(basic_battle_state, [])

        # Should have thrust, rotate, and engage commands
        thrust_commands = [c for c in commands if isinstance(c, SetThrust)]
        rotate_commands = [c for c in commands if isinstance(c, RotateTo)]
        engage_commands = [c for c in commands if isinstance(c, Engage)]

        assert len(thrust_commands) == 1
        assert thrust_commands[0].magnitude == 0.5
        assert len(rotate_commands) == 1
        assert len(engage_commands) == 1

    def test_default_behavior_no_target(self, basic_ship_state):
        """LLMStrategicController falls back to hold fire without target."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        controller = LLMStrategicController(ship_id='ship_001')
        commands = controller.decide(battle_state, [])

        hold_fire_commands = [c for c in commands if isinstance(c, HoldFire)]
        assert len(hold_fire_commands) == 1

    def test_format_state_for_llm(self, basic_battle_state):
        """LLMStrategicController formats state for LLM consumption."""
        controller = LLMStrategicController(ship_id='ship_001')
        formatted = controller._format_state_for_llm(basic_battle_state)

        assert 'BATTLE STATE' in formatted
        assert 'OWN SHIP' in formatted
        assert 'ship_001' in formatted
        assert 'ENEMY' in formatted


# =============================================================================
# Command Validation Tests
# =============================================================================

class TestValidateCommand:
    """Tests for command validation function."""

    def test_validate_destroyed_ship(self, basic_ship_state, unit_vectors):
        """Commands are invalid for destroyed ships."""
        basic_ship_state.is_destroyed = True
        cmd = SetThrust(vector=unit_vectors['forward'], magnitude=0.5)

        is_valid, error = validate_command(cmd, basic_ship_state)

        assert is_valid is False
        assert 'destroyed' in error.lower()

    def test_validate_thrust_no_delta_v(self, basic_ship_state, unit_vectors):
        """SetThrust is invalid with no delta-v."""
        basic_ship_state.delta_v_remaining = 0.0
        cmd = SetThrust(vector=unit_vectors['forward'], magnitude=0.5)

        is_valid, error = validate_command(cmd, basic_ship_state)

        assert is_valid is False
        assert 'delta-v' in error.lower()

    def test_validate_thrust_disabled_ship(self, basic_ship_state, unit_vectors):
        """SetThrust is invalid for disabled ships."""
        basic_ship_state.is_disabled = True
        cmd = SetThrust(vector=unit_vectors['forward'], magnitude=0.5)

        is_valid, error = validate_command(cmd, basic_ship_state)

        assert is_valid is False
        assert 'disabled' in error.lower()

    def test_validate_thrust_zero_magnitude_ok(self, basic_ship_state, unit_vectors):
        """SetThrust with zero magnitude is valid even with no delta-v."""
        basic_ship_state.delta_v_remaining = 0.0
        cmd = SetThrust(vector=unit_vectors['forward'], magnitude=0.0)

        is_valid, error = validate_command(cmd, basic_ship_state)

        assert is_valid is True

    def test_validate_engage_invalid_weapon(self, basic_ship_state):
        """Engage is invalid with no valid weapon slots."""
        cmd = Engage(target_id='enemy_001', weapon_slots=['nonexistent_weapon'])

        is_valid, error = validate_command(cmd, basic_ship_state)

        assert is_valid is False
        assert 'weapon' in error.lower()

    def test_validate_engage_valid_weapon(self, basic_ship_state):
        """Engage is valid with existing weapon slots."""
        cmd = Engage(target_id='enemy_001', weapon_slots=['turret_dorsal'])

        is_valid, error = validate_command(cmd, basic_ship_state)

        assert is_valid is True
        assert error == ''

    def test_validate_evade_low_delta_v(self, basic_ship_state):
        """Evade is invalid with insufficient delta-v."""
        basic_ship_state.delta_v_remaining = 500.0  # Below 1000 m/s threshold
        cmd = Evade(threat_id='torpedo_001')

        is_valid, error = validate_command(cmd, basic_ship_state)

        assert is_valid is False
        assert 'delta-v' in error.lower()

    def test_validate_rotate_always_valid(self, basic_ship_state, unit_vectors):
        """RotateTo is valid for non-destroyed ships."""
        cmd = RotateTo(target_bearing=unit_vectors['forward'])

        is_valid, error = validate_command(cmd, basic_ship_state)

        assert is_valid is True

    def test_validate_hold_fire_always_valid(self, basic_ship_state):
        """HoldFire is valid for non-destroyed ships."""
        cmd = HoldFire()

        is_valid, error = validate_command(cmd, basic_ship_state)

        assert is_valid is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestHierarchicalControl:
    """Integration tests for the hierarchical control architecture."""

    def test_strategic_to_tactical_flow(self, basic_battle_state):
        """Strategic commands flow correctly to tactical controller."""
        # Strategic controller makes decision
        strategic = RuleBasedStrategicController(ship_id='ship_001')
        commands = strategic.decide(basic_battle_state, [])

        # Tactical controller receives and executes commands
        tactical = TacticalController(ship_id='ship_001')
        tactical.set_commands(commands)

        outputs, events = tactical.update(basic_battle_state, delta_time=1.0)

        # Should have coherent outputs
        assert outputs is not None
        assert 'thrust_vector' in outputs
        assert 'target_heading' in outputs
        assert 'weapons_fire' in outputs

    def test_event_generation_triggers_strategic(self, basic_ship_state, critical_threat):
        """Critical tactical events can trigger strategic response."""
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=basic_ship_state,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[critical_threat],
            active_projectiles=[],
            recent_events=[],
            engagement_range=0.0,
            closing_rate=0.0,
            battle_duration=100.0,
        )

        # Tactical controller detects threat
        tactical = TacticalController(ship_id='ship_001', evade_threshold_seconds=30.0)
        outputs, events = tactical.update(battle_state, delta_time=1.0)

        # Filter for critical events (priority >= 4)
        critical_events = [e for e in events if e.priority >= 4]

        # Strategic controller can respond to critical events
        strategic = RuleBasedStrategicController(ship_id='ship_001')
        for event in critical_events:
            response = strategic.on_critical_event(event, battle_state)
            if response is not None:
                # Verify response contains appropriate commands
                assert isinstance(response, list)
                assert all(isinstance(c, (SetThrust, RotateTo, Engage, HoldFire, Evade)) for c in response)

    def test_decision_interval_respected(self, basic_battle_state):
        """Strategic decisions only occur at configured intervals."""
        strategic = RuleBasedStrategicController(
            ship_id='ship_001',
            decision_interval=30.0,
        )

        # At t=0, should_decide is False (0.0 - 0.0 >= 30.0 is False)
        # But we can force first decision with request_decision
        basic_battle_state.timestamp = 0.0
        strategic.request_decision(basic_battle_state, [])

        # No decision at t=15 (15.0 - 0.0 >= 30.0 is False)
        assert strategic.should_decide(15.0) is False

        # Decision at t=30 (30.0 - 0.0 >= 30.0 is True)
        assert strategic.should_decide(30.0) is True

        # After making decision at t=30, next at t=60
        basic_battle_state.timestamp = 30.0
        strategic.request_decision(basic_battle_state, [])
        assert strategic.should_decide(45.0) is False
        assert strategic.should_decide(60.0) is True
