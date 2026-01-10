#!/usr/bin/env python3
"""
Tests for the fire control system.

Tests:
- Hit probability calculation
- Command schemas (HelmOrder, WeaponsOrder, TacticalOrder)
- Weapons officer AI
- Engagement envelope calculation
"""

import math
import pytest
from dataclasses import dataclass
from typing import Optional

from src.firecontrol import (
    FiringSolution,
    calculate_hit_probability,
    calculate_engagement_envelope,
    HelmCommand,
    WeaponsCommand,
    TacticalPosture,
    HelmOrder,
    WeaponsOrder,
    TacticalOrder,
    WeaponsOfficer,
    WeaponStatus,
)
from src.physics import Vector3D
from src.geometry import ShipGeometry
from src.combat import HitLocation


class TestFiringSolution:
    """Tests for FiringSolution dataclass."""

    def test_is_optimal_above_threshold(self):
        """is_optimal returns True when probability >= 30%."""
        solution = FiringSolution(
            can_fire=True,
            hit_probability=0.35,
            time_of_flight_s=5.0,
            predicted_range_km=50.0,
            target_aspect=HitLocation.NOSE,
            recommendation="MARGINAL"
        )
        assert solution.is_optimal is True

    def test_is_optimal_at_threshold(self):
        """is_optimal returns True when probability == 30%."""
        solution = FiringSolution(
            can_fire=True,
            hit_probability=0.30,
            time_of_flight_s=5.0,
            predicted_range_km=50.0,
            target_aspect=HitLocation.NOSE,
            recommendation="MARGINAL"
        )
        assert solution.is_optimal is True

    def test_is_optimal_below_threshold(self):
        """is_optimal returns False when probability < 30%."""
        solution = FiringSolution(
            can_fire=True,
            hit_probability=0.25,
            time_of_flight_s=5.0,
            predicted_range_km=50.0,
            target_aspect=HitLocation.NOSE,
            recommendation="POOR"
        )
        assert solution.is_optimal is False

    def test_is_good_above_threshold(self):
        """is_good returns True when probability >= 50%."""
        solution = FiringSolution(
            can_fire=True,
            hit_probability=0.55,
            time_of_flight_s=3.0,
            predicted_range_km=30.0,
            target_aspect=HitLocation.NOSE,
            recommendation="GOOD"
        )
        assert solution.is_good is True

    def test_is_good_below_threshold(self):
        """is_good returns False when probability < 50%."""
        solution = FiringSolution(
            can_fire=True,
            hit_probability=0.45,
            time_of_flight_s=5.0,
            predicted_range_km=50.0,
            target_aspect=HitLocation.NOSE,
            recommendation="MARGINAL"
        )
        assert solution.is_good is False


class TestCalculateHitProbability:
    """Tests for hit probability calculation."""

    @pytest.fixture
    def standard_geometry(self):
        """Standard destroyer-sized target geometry."""
        # ShipGeometry uses length_m, radius_m, nose_cone_length_m, engine_section_length_m
        return ShipGeometry(
            length_m=125.0,
            radius_m=15.6,  # ~length/8
            nose_cone_length_m=25.0,  # ~20% of length
            engine_section_length_m=18.75  # ~15% of length
        )

    def test_point_blank_guaranteed_hit(self, standard_geometry):
        """At point blank range (<1m), hit is guaranteed."""
        solution = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(0.5, 0, 0),  # 0.5 meters away
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        )
        assert solution.can_fire is True
        assert solution.hit_probability == 1.0
        assert solution.time_of_flight_s == 0.0
        assert "POINT BLANK" in solution.recommendation

    def test_close_range_high_probability(self, standard_geometry):
        """Close range (~10km) should have high hit probability."""
        solution = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(10_000, 0, 0),  # 10 km
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        )
        assert solution.can_fire is True
        assert solution.hit_probability > 0.3  # Should be good shot

    def test_long_range_low_probability(self, standard_geometry):
        """Long range (~500km) should have lower hit probability."""
        solution = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(500_000, 0, 0),  # 500 km
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        )
        assert solution.can_fire is True
        assert solution.hit_probability < 0.3  # Should be poor shot

    def test_probability_decreases_with_range(self, standard_geometry):
        """Hit probability should decrease as range increases."""
        prob_10km = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(10_000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        ).hit_probability

        prob_50km = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(50_000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        ).hit_probability

        prob_100km = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(100_000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        ).hit_probability

        assert prob_10km > prob_50km > prob_100km

    def test_target_outrunning_projectile(self, standard_geometry):
        """Can't hit target moving away faster than projectile."""
        solution = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(100_000, 0, 0),  # 100 km
            target_velocity=Vector3D(15_000, 0, 0),  # 15 km/s away
            target_geometry=standard_geometry,
            target_forward=Vector3D(1, 0, 0),
            muzzle_velocity_kps=10.0  # 10 km/s muzzle velocity
        )
        assert solution.can_fire is False
        assert solution.hit_probability == 0.0
        assert "OUTRUNNING" in solution.recommendation

    def test_closing_targets_easier_to_hit(self, standard_geometry):
        """Targets closing on shooter should be easier to hit."""
        # Target moving toward shooter
        prob_closing = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(100_000, 0, 0),
            target_velocity=Vector3D(-5_000, 0, 0),  # 5 km/s toward
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        ).hit_probability

        # Stationary target
        prob_stationary = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(100_000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        ).hit_probability

        # Closing target should have equal or better probability
        # (shorter time of flight)
        assert prob_closing >= prob_stationary * 0.9  # Allow some tolerance

    def test_evasion_reduces_probability(self, standard_geometry):
        """Evading targets should be harder to hit."""
        prob_normal = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(50_000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0,
            target_is_evading=False
        ).hit_probability

        prob_evading = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(50_000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0,
            target_is_evading=True
        ).hit_probability

        assert prob_evading < prob_normal

    def test_nose_aspect(self, standard_geometry):
        """Target facing shooter should show NOSE aspect."""
        solution = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(50_000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),  # Facing toward shooter
            muzzle_velocity_kps=10.0
        )
        assert solution.target_aspect == HitLocation.NOSE

    def test_tail_aspect(self, standard_geometry):
        """Target facing away should show TAIL aspect."""
        solution = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(50_000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(1, 0, 0),  # Facing away from shooter
            muzzle_velocity_kps=10.0
        )
        assert solution.target_aspect == HitLocation.TAIL

    def test_lateral_aspect(self, standard_geometry):
        """Target broadside to shooter should show LATERAL aspect."""
        solution = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(50_000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(0, 1, 0),  # Perpendicular
            muzzle_velocity_kps=10.0
        )
        assert solution.target_aspect == HitLocation.LATERAL

    def test_recommendations(self, standard_geometry):
        """Different probability levels should give different recommendations."""
        # High probability - excellent
        solution_excellent = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(5_000, 0, 0),  # Very close
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        )
        assert "EXCELLENT" in solution_excellent.recommendation or \
               "GOOD" in solution_excellent.recommendation

        # Low probability - poor/hold
        solution_poor = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(1_000_000, 0, 0),  # 1000 km
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        )
        assert "POOR" in solution_poor.recommendation or \
               "HOLD" in solution_poor.recommendation

    def test_probability_clamped(self, standard_geometry):
        """Probability should be clamped between 0.01 and 0.99."""
        # Very close - should not exceed 0.99
        solution_close = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(100, 0, 0),  # 100 meters
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        )
        assert solution_close.hit_probability <= 0.99

        # Very far - should not go below 0.01
        solution_far = calculate_hit_probability(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_position=Vector3D(10_000_000, 0, 0),  # 10,000 km
            target_velocity=Vector3D(0, 0, 0),
            target_geometry=standard_geometry,
            target_forward=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=10.0
        )
        assert solution_far.hit_probability >= 0.01


class TestEngagementEnvelope:
    """Tests for engagement envelope calculation."""

    def test_envelope_structure(self):
        """Envelope should have expected keys."""
        envelope = calculate_engagement_envelope(
            weapon_range_km=1000.0,
            muzzle_velocity_kps=10.0,
            target_size_m=60.0
        )
        assert 'optimal_range_km' in envelope
        assert 'good_range_km' in envelope
        assert 'max_range_km' in envelope
        assert 'brackets' in envelope

    def test_envelope_max_range(self):
        """Max range should match weapon range."""
        envelope = calculate_engagement_envelope(
            weapon_range_km=500.0,
            muzzle_velocity_kps=10.0
        )
        assert envelope['max_range_km'] == 500.0

    def test_envelope_brackets_sorted(self):
        """Range brackets should be in ascending order."""
        envelope = calculate_engagement_envelope(
            weapon_range_km=1000.0,
            muzzle_velocity_kps=10.0
        )
        ranges = [b['range_km'] for b in envelope['brackets']]
        assert ranges == sorted(ranges)

    def test_envelope_probability_decreases(self):
        """Hit probability should decrease with range."""
        envelope = calculate_engagement_envelope(
            weapon_range_km=1000.0,
            muzzle_velocity_kps=10.0
        )
        probs = [b['hit_probability'] for b in envelope['brackets']]
        # Each probability should be less than or equal to the previous
        for i in range(1, len(probs)):
            assert probs[i] <= probs[i-1]


class TestHelmCommands:
    """Tests for helm command enums and orders."""

    def test_all_helm_commands_exist(self):
        """All expected helm commands should exist."""
        commands = [
            HelmCommand.INTERCEPT,
            HelmCommand.FACE_TARGET,
            HelmCommand.EVADE,
            HelmCommand.BRAKE,
            HelmCommand.HOLD_COURSE,
            HelmCommand.ROTATE_TO,
            HelmCommand.PURSUIT
        ]
        assert len(commands) == len(HelmCommand)

    def test_helm_order_creation(self):
        """HelmOrder should be creatable with default values."""
        order = HelmOrder(command=HelmCommand.INTERCEPT)
        assert order.command == HelmCommand.INTERCEPT
        assert order.throttle == 1.0
        assert order.target_id is None
        assert order.direction is None
        assert order.duration == 0.0
        assert order.evasion_intensity == 0.5

    def test_helm_order_with_target(self):
        """HelmOrder should accept target_id."""
        order = HelmOrder(
            command=HelmCommand.INTERCEPT,
            target_id="enemy-1",
            throttle=0.8
        )
        assert order.target_id == "enemy-1"
        assert order.throttle == 0.8

    def test_helm_order_evade(self):
        """Evade order should use evasion_intensity."""
        order = HelmOrder(
            command=HelmCommand.EVADE,
            evasion_intensity=0.9
        )
        assert order.command == HelmCommand.EVADE
        assert order.evasion_intensity == 0.9


class TestWeaponsCommands:
    """Tests for weapons command enums and orders."""

    def test_all_weapons_commands_exist(self):
        """All expected weapons commands should exist."""
        commands = [
            WeaponsCommand.FIRE_IMMEDIATE,
            WeaponsCommand.FIRE_WHEN_OPTIMAL,
            WeaponsCommand.FIRE_AT_RANGE,
            WeaponsCommand.HOLD_FIRE,
            WeaponsCommand.FREE_FIRE
        ]
        assert len(commands) == len(WeaponsCommand)

    def test_weapons_order_creation(self):
        """WeaponsOrder should be creatable with default values."""
        order = WeaponsOrder(command=WeaponsCommand.FIRE_WHEN_OPTIMAL)
        assert order.command == WeaponsCommand.FIRE_WHEN_OPTIMAL
        assert order.weapon_slot == "all"
        assert order.target_id is None
        assert order.min_hit_probability == 0.3
        assert order.max_range_km == 0.0
        assert order.conserve_ammo is False

    def test_weapons_order_fire_at_range(self):
        """Fire at range order should use max_range_km."""
        order = WeaponsOrder(
            command=WeaponsCommand.FIRE_AT_RANGE,
            weapon_slot="weapon_0",
            max_range_km=100.0
        )
        assert order.max_range_km == 100.0
        assert order.weapon_slot == "weapon_0"

    def test_weapons_order_conserve_ammo(self):
        """Conserve ammo flag should be settable."""
        order = WeaponsOrder(
            command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
            conserve_ammo=True,
            min_hit_probability=0.5
        )
        assert order.conserve_ammo is True
        assert order.min_hit_probability == 0.5


class TestTacticalPosture:
    """Tests for tactical posture enum."""

    def test_all_postures_exist(self):
        """All expected tactical postures should exist."""
        postures = [
            TacticalPosture.AGGRESSIVE,
            TacticalPosture.BALANCED,
            TacticalPosture.DEFENSIVE,
            TacticalPosture.WITHDRAW
        ]
        assert len(postures) == len(TacticalPosture)


class TestTacticalOrder:
    """Tests for tactical order dataclass."""

    def test_tactical_order_defaults(self):
        """TacticalOrder should have sensible defaults."""
        order = TacticalOrder()
        assert order.posture == TacticalPosture.BALANCED
        assert order.primary_target is None
        assert order.helm_order is None
        assert order.weapons_orders == []
        assert order.priority == 0

    def test_tactical_order_full(self):
        """TacticalOrder should accept all parameters."""
        helm = HelmOrder(command=HelmCommand.INTERCEPT, target_id="enemy-1")
        weapons = WeaponsOrder(command=WeaponsCommand.FIRE_WHEN_OPTIMAL)

        order = TacticalOrder(
            posture=TacticalPosture.AGGRESSIVE,
            primary_target="enemy-1",
            helm_order=helm,
            weapons_orders=[weapons],
            priority=10
        )

        assert order.posture == TacticalPosture.AGGRESSIVE
        assert order.primary_target == "enemy-1"
        assert order.helm_order == helm
        assert len(order.weapons_orders) == 1
        assert order.priority == 10


class TestWeaponsOfficer:
    """Tests for the weapons officer AI."""

    @pytest.fixture
    def weapons_officer(self):
        """Create a standard weapons officer."""
        return WeaponsOfficer(
            min_probability_threshold=0.3,
            conserve_ammo_threshold=0.5,
            max_ammo_reserve_percent=0.2
        )

    @pytest.fixture
    def good_solution(self):
        """A firing solution with good hit probability."""
        return FiringSolution(
            can_fire=True,
            hit_probability=0.55,
            time_of_flight_s=5.0,
            predicted_range_km=50.0,
            target_aspect=HitLocation.NOSE,
            recommendation="GOOD SHOT - RECOMMEND FIRE"
        )

    @pytest.fixture
    def poor_solution(self):
        """A firing solution with poor hit probability."""
        return FiringSolution(
            can_fire=True,
            hit_probability=0.15,
            time_of_flight_s=20.0,
            predicted_range_km=200.0,
            target_aspect=HitLocation.LATERAL,
            recommendation="POOR - CLOSE RANGE OR WAIT"
        )

    @pytest.fixture
    def cannot_fire_solution(self):
        """A firing solution where weapon cannot fire."""
        return FiringSolution(
            can_fire=False,
            hit_probability=0.0,
            time_of_flight_s=float('inf'),
            predicted_range_km=500.0,
            target_aspect=HitLocation.TAIL,
            recommendation="TARGET OUTRUNNING PROJECTILE"
        )

    def test_set_and_get_order(self, weapons_officer):
        """Should be able to set and retrieve orders."""
        order = WeaponsOrder(
            command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
            weapon_slot="weapon_0",
            min_hit_probability=0.5
        )
        weapons_officer.set_order(order)
        retrieved = weapons_officer.get_order("weapon_0")
        assert retrieved == order

    def test_default_order(self, weapons_officer):
        """Setting 'all' slot should create default order."""
        order = WeaponsOrder(
            command=WeaponsCommand.HOLD_FIRE,
            weapon_slot="all"
        )
        weapons_officer.set_order(order)
        # Should retrieve default for unknown slot
        retrieved = weapons_officer.get_order("weapon_99")
        assert retrieved.command == WeaponsCommand.HOLD_FIRE

    def test_evaluate_hold_fire(self, weapons_officer, good_solution):
        """HOLD_FIRE should always return False."""
        order = WeaponsOrder(command=WeaponsCommand.HOLD_FIRE)
        should_fire, reason = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=good_solution,
            order=order
        )
        assert should_fire is False
        assert "HOLD FIRE" in reason

    def test_evaluate_fire_immediate(self, weapons_officer, poor_solution):
        """FIRE_IMMEDIATE should fire regardless of probability."""
        order = WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE)
        should_fire, reason = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=poor_solution,
            order=order
        )
        assert should_fire is True
        assert "IMMEDIATE" in reason

    def test_evaluate_free_fire_in_range(self, weapons_officer, poor_solution):
        """FREE_FIRE should fire if probability >= 10%."""
        order = WeaponsOrder(command=WeaponsCommand.FREE_FIRE)
        should_fire, reason = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=poor_solution,  # 15% probability
            order=order
        )
        assert should_fire is True
        assert "FREE FIRE" in reason

    def test_evaluate_fire_at_range_in_range(self, weapons_officer, good_solution):
        """FIRE_AT_RANGE should fire if predicted range <= max_range."""
        order = WeaponsOrder(
            command=WeaponsCommand.FIRE_AT_RANGE,
            max_range_km=100.0
        )
        should_fire, reason = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=good_solution,  # 50 km predicted range
            order=order
        )
        assert should_fire is True
        assert "in range" in reason.lower()

    def test_evaluate_fire_at_range_out_of_range(self, weapons_officer, poor_solution):
        """FIRE_AT_RANGE should not fire if predicted range > max_range."""
        order = WeaponsOrder(
            command=WeaponsCommand.FIRE_AT_RANGE,
            max_range_km=100.0
        )
        should_fire, reason = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=poor_solution,  # 200 km predicted range
            order=order
        )
        assert should_fire is False
        assert "out of range" in reason.lower()

    def test_evaluate_fire_when_optimal_good_shot(self, weapons_officer, good_solution):
        """FIRE_WHEN_OPTIMAL should fire when probability >= threshold."""
        order = WeaponsOrder(
            command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
            min_hit_probability=0.3
        )
        should_fire, reason = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=good_solution,  # 55% probability
            order=order
        )
        assert should_fire is True

    def test_evaluate_fire_when_optimal_poor_shot(self, weapons_officer, poor_solution):
        """FIRE_WHEN_OPTIMAL should not fire when probability < threshold."""
        order = WeaponsOrder(
            command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
            min_hit_probability=0.3
        )
        should_fire, reason = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=poor_solution,  # 15% probability
            order=order
        )
        assert should_fire is False
        assert "too low" in reason.lower()

    def test_evaluate_cannot_fire(self, weapons_officer, cannot_fire_solution):
        """Should not fire when solution says cannot fire."""
        order = WeaponsOrder(command=WeaponsCommand.FIRE_IMMEDIATE)
        should_fire, reason = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=cannot_fire_solution,
            order=order
        )
        assert should_fire is False
        assert "OUTRUNNING" in reason

    def test_evaluate_low_ammo_more_selective(self, weapons_officer):
        """When ammo is low, should be more selective."""
        # Solution with marginal probability (35%)
        marginal_solution = FiringSolution(
            can_fire=True,
            hit_probability=0.35,
            time_of_flight_s=10.0,
            predicted_range_km=100.0,
            target_aspect=HitLocation.NOSE,
            recommendation="MARGINAL"
        )

        order = WeaponsOrder(
            command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
            min_hit_probability=0.3
        )

        # Full ammo - should fire
        should_fire_full, _ = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=marginal_solution,
            order=order
        )
        assert should_fire_full is True

        # Low ammo (30%) - should not fire due to raised threshold
        should_fire_low, _ = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=30,
            weapon_magazine=100,
            solution=marginal_solution,
            order=order
        )
        assert should_fire_low is False

    def test_evaluate_ammo_critical_reserve(self, weapons_officer, good_solution):
        """Should not fire when at ammo reserve level."""
        order = WeaponsOrder(
            command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
            min_hit_probability=0.3
        )
        # At 20% ammo (reserve level)
        should_fire, reason = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=20,
            weapon_magazine=100,
            solution=good_solution,
            order=order
        )
        assert should_fire is False
        assert "AMMO CRITICAL" in reason

    def test_evaluate_conserve_ammo_flag(self, weapons_officer):
        """Conserve ammo flag should raise threshold."""
        marginal_solution = FiringSolution(
            can_fire=True,
            hit_probability=0.45,
            time_of_flight_s=8.0,
            predicted_range_km=80.0,
            target_aspect=HitLocation.NOSE,
            recommendation="MARGINAL"
        )

        # Without conserve flag - should fire (45% >= 30%)
        order_normal = WeaponsOrder(
            command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
            min_hit_probability=0.3,
            conserve_ammo=False
        )
        should_fire_normal, _ = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=marginal_solution,
            order=order_normal
        )
        assert should_fire_normal is True

        # With conserve flag - should not fire (45% < 50%)
        order_conserve = WeaponsOrder(
            command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
            min_hit_probability=0.3,
            conserve_ammo=True
        )
        should_fire_conserve, _ = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=marginal_solution,
            order=order_conserve
        )
        assert should_fire_conserve is False

    def test_default_order_when_none_set(self, weapons_officer, good_solution):
        """Without explicit order, should use FIRE_WHEN_OPTIMAL."""
        # No order set
        should_fire, _ = weapons_officer.evaluate_shot(
            weapon_slot="weapon_0",
            weapon_ammo=100,
            weapon_magazine=100,
            solution=good_solution,
            order=None
        )
        assert should_fire is True  # Good solution should fire with default


class TestWeaponsOfficerIntegration:
    """Integration tests for weapons officer with simulated combat."""

    @pytest.fixture
    def mock_weapon_state(self):
        """Create a mock weapon state for testing."""
        @dataclass
        class MockWeapon:
            name: str = "Coilgun"
            range_km: float = 1000.0
            muzzle_velocity_kps: float = 10.0
            magazine: int = 100

        @dataclass
        class MockWeaponState:
            weapon: MockWeapon = None
            ammo_remaining: int = 100
            cooldown_remaining: float = 0.0

            def __post_init__(self):
                if self.weapon is None:
                    self.weapon = MockWeapon()

            def can_fire(self) -> bool:
                return self.cooldown_remaining <= 0 and self.ammo_remaining > 0

        return MockWeaponState

    @pytest.fixture
    def mock_target(self):
        """Create a mock target ship."""
        @dataclass
        class MockTarget:
            ship_id: str = "target-1"
            position: Vector3D = None
            velocity: Vector3D = None
            forward: Vector3D = None
            geometry: ShipGeometry = None
            is_evading: bool = False

            def __post_init__(self):
                if self.position is None:
                    self.position = Vector3D(50_000, 0, 0)
                if self.velocity is None:
                    self.velocity = Vector3D(0, 0, 0)
                if self.forward is None:
                    self.forward = Vector3D(-1, 0, 0)
                if self.geometry is None:
                    self.geometry = ShipGeometry(
                        length_m=125.0,
                        radius_m=15.6,
                        nose_cone_length_m=25.0,
                        engine_section_length_m=18.75
                    )

        return MockTarget

    def test_get_fire_commands_no_targets(self, mock_weapon_state):
        """Should return empty list when no targets."""
        officer = WeaponsOfficer()
        weapons = {"weapon_0": mock_weapon_state()}

        commands = officer.get_fire_commands(
            ship_position=Vector3D(0, 0, 0),
            ship_velocity=Vector3D(0, 0, 0),
            weapons=weapons,
            targets=[]
        )
        assert commands == []

    def test_get_fire_commands_with_valid_target(self, mock_weapon_state, mock_target):
        """Should return fire command for valid target."""
        officer = WeaponsOfficer(min_probability_threshold=0.1)  # Low threshold
        weapons = {"weapon_0": mock_weapon_state()}
        targets = [mock_target()]

        commands = officer.get_fire_commands(
            ship_position=Vector3D(0, 0, 0),
            ship_velocity=Vector3D(0, 0, 0),
            weapons=weapons,
            targets=targets
        )

        assert len(commands) == 1
        assert commands[0]['type'] == 'fire_at'
        assert commands[0]['weapon_slot'] == 'weapon_0'
        assert commands[0]['target_id'] == 'target-1'
        assert 'hit_probability' in commands[0]

    def test_get_fire_commands_skips_pd(self, mock_weapon_state, mock_target):
        """Should skip point defense weapons."""
        officer = WeaponsOfficer(min_probability_threshold=0.1)
        weapons = {
            "weapon_0": mock_weapon_state(),
            "pd_0": mock_weapon_state()  # Point defense
        }
        targets = [mock_target()]

        commands = officer.get_fire_commands(
            ship_position=Vector3D(0, 0, 0),
            ship_velocity=Vector3D(0, 0, 0),
            weapons=weapons,
            targets=targets
        )

        # Only weapon_0, not pd_0
        assert len(commands) == 1
        assert commands[0]['weapon_slot'] == 'weapon_0'

    def test_get_fire_commands_respects_order(self, mock_weapon_state, mock_target):
        """Should respect HOLD_FIRE order."""
        officer = WeaponsOfficer()
        officer.set_order(WeaponsOrder(
            command=WeaponsCommand.HOLD_FIRE,
            weapon_slot="all"
        ))

        weapons = {"weapon_0": mock_weapon_state()}
        targets = [mock_target()]

        commands = officer.get_fire_commands(
            ship_position=Vector3D(0, 0, 0),
            ship_velocity=Vector3D(0, 0, 0),
            weapons=weapons,
            targets=targets
        )

        assert commands == []

    def test_get_fire_commands_out_of_range(self, mock_weapon_state, mock_target):
        """Should not fire at target out of weapon range."""
        officer = WeaponsOfficer(min_probability_threshold=0.01)

        # Weapon with very short range
        weapon = mock_weapon_state()
        weapon.weapon.range_km = 10.0  # 10km range
        weapons = {"weapon_0": weapon}

        # Target at 50km
        target = mock_target()
        target.position = Vector3D(50_000, 0, 0)

        commands = officer.get_fire_commands(
            ship_position=Vector3D(0, 0, 0),
            ship_velocity=Vector3D(0, 0, 0),
            weapons=weapons,
            targets=[target]
        )

        assert commands == []

    def test_get_fire_commands_weapon_not_ready(self, mock_weapon_state, mock_target):
        """Should not fire weapons that aren't ready."""
        officer = WeaponsOfficer(min_probability_threshold=0.01)

        weapon = mock_weapon_state()
        weapon.cooldown_remaining = 5.0  # On cooldown
        weapons = {"weapon_0": weapon}

        commands = officer.get_fire_commands(
            ship_position=Vector3D(0, 0, 0),
            ship_velocity=Vector3D(0, 0, 0),
            weapons=weapons,
            targets=[mock_target()]
        )

        assert commands == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
