"""
Integration tests for the AI Commanders Space Battle Simulator.

These scenario-based tests verify that the physics, combat, and command modules
work together correctly as a complete system.

Run with: python -m pytest tests/test_scenarios.py -v
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from src.physics import (
    Vector3D,
    ShipState,
    create_ship_state_from_specs,
    propagate_state,
    propagate_trajectory,
    time_to_rotate,
    angular_acceleration_from_torque,
    tsiolkovsky_delta_v,
    EXHAUST_VELOCITY_KPS,
    G_STANDARD,
)
from src.combat import (
    Armor,
    CombatResolver,
    HitLocation,
    HitResult,
    ShipArmor,
    Weapon,
    load_fleet_data,
    create_weapon_from_fleet_data,
    create_ship_armor_from_fleet_data,
)
from src.command import (
    BattleState,
    ShipState as CommandShipState,
    ThreatInfo,
    ProjectileInfo,
    TacticalController,
    TacticalEvent,
    TacticalEventType,
    RuleBasedStrategicController,
    SetThrust,
    RotateTo,
    Engage,
    HoldFire,
    Evade,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def fleet_data() -> dict:
    """Load fleet data from the data directory."""
    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    if not data_path.exists():
        pytest.skip("Fleet data file not found")
    return load_fleet_data(data_path)


@pytest.fixture
def seeded_rng() -> random.Random:
    """Create a seeded random number generator for deterministic tests."""
    return random.Random(42)


@pytest.fixture
def seeded_resolver(seeded_rng) -> CombatResolver:
    """Create a combat resolver with fixed seed."""
    return CombatResolver(rng=seeded_rng)


# =============================================================================
# Helper Functions
# =============================================================================

def create_ship_from_fleet(
    fleet_data: dict,
    ship_type: str,
    position: Vector3D,
    velocity: Vector3D,
    forward: Vector3D,
) -> ShipState:
    """Create a physics ShipState from fleet data."""
    ship_data = fleet_data["ships"][ship_type]
    perf = ship_data["performance"]
    hull = ship_data["hull"]
    propulsion = ship_data["propulsion"]
    attitude = ship_data["attitude_control"]

    return create_ship_state_from_specs(
        wet_mass_tons=perf["max_wet_mass_tons"],
        dry_mass_tons=perf["max_dry_mass_tons"],
        length_m=hull["length_m"],
        thrust_mn=propulsion["drive"]["thrust_mn"],
        exhaust_velocity_kps=propulsion["drive"]["exhaust_velocity_kps"],
        position=position,
        velocity=velocity,
        forward=forward,
    )


def calculate_closure_rate(
    ship_a_pos: Vector3D,
    ship_a_vel: Vector3D,
    ship_b_pos: Vector3D,
    ship_b_vel: Vector3D,
) -> float:
    """
    Calculate the closure rate between two ships.

    Positive value means ships are approaching, negative means separating.
    """
    relative_pos = ship_b_pos - ship_a_pos
    relative_vel = ship_b_vel - ship_a_vel

    distance = relative_pos.magnitude
    if distance == 0:
        return 0.0

    # Closure rate is negative of the radial component of relative velocity
    unit_direction = relative_pos.normalized()
    closure_rate = -relative_vel.dot(unit_direction)

    return closure_rate


def time_to_engagement_range(
    ship_a_pos: Vector3D,
    ship_a_vel: Vector3D,
    ship_b_pos: Vector3D,
    ship_b_vel: Vector3D,
    engagement_range_m: float,
) -> float:
    """
    Calculate time until ships reach engagement range.

    Returns float('inf') if ships are separating or won't reach range.
    """
    distance = (ship_b_pos - ship_a_pos).magnitude
    closure_rate = calculate_closure_rate(ship_a_pos, ship_a_vel, ship_b_pos, ship_b_vel)

    if closure_rate <= 0:
        # Ships separating
        return float('inf')

    if distance <= engagement_range_m:
        # Already in range
        return 0.0

    return (distance - engagement_range_m) / closure_rate


def create_command_ship_state(
    ship_id: str,
    physics_state: ShipState,
    weapons_status: dict,
) -> CommandShipState:
    """Create a command module ShipState from a physics ShipState."""
    return CommandShipState(
        ship_id=ship_id,
        position=np.array([physics_state.position.x, physics_state.position.y, physics_state.position.z]),
        velocity=np.array([physics_state.velocity.x, physics_state.velocity.y, physics_state.velocity.z]),
        heading=np.array([physics_state.forward.x, physics_state.forward.y, physics_state.forward.z]),
        angular_velocity=np.array([
            physics_state.angular_velocity.x,
            physics_state.angular_velocity.y,
            physics_state.angular_velocity.z,
        ]),
        hull_hp={"main": (100.0, 100.0)},
        heat_percentage=20.0,
        battery_percentage=100.0,
        delta_v_remaining=physics_state.remaining_delta_v_ms(),
        weapons_status=weapons_status,
    )


# =============================================================================
# Scenario 1: Intercept Scenario
# =============================================================================

class TestInterceptScenario:
    """
    Test scenario: Two ships approaching each other.

    Tests closure rate calculation and time to engagement range.
    """

    def test_closure_rate_head_on_approach(self, fleet_data):
        """Test closure rate when ships approach head-on."""
        # Ship A moving +X at 10 km/s
        ship_a_pos = Vector3D(0, 0, 0)
        ship_a_vel = Vector3D(10_000, 0, 0)  # 10 km/s

        # Ship B moving -X at 10 km/s (toward ship A)
        ship_b_pos = Vector3D(1_000_000, 0, 0)  # 1000 km away
        ship_b_vel = Vector3D(-10_000, 0, 0)  # 10 km/s toward A

        closure_rate = calculate_closure_rate(ship_a_pos, ship_a_vel, ship_b_pos, ship_b_vel)

        # Combined closure rate should be 20 km/s = 20,000 m/s
        assert abs(closure_rate - 20_000) < 1, f"Expected 20000 m/s, got {closure_rate}"

    def test_closure_rate_perpendicular(self, fleet_data):
        """Test closure rate when ships move perpendicular to line of sight."""
        # Ship A stationary
        ship_a_pos = Vector3D(0, 0, 0)
        ship_a_vel = Vector3D(0, 0, 0)

        # Ship B moving perpendicular to the line between them
        ship_b_pos = Vector3D(1_000_000, 0, 0)  # 1000 km away along X
        ship_b_vel = Vector3D(0, 10_000, 0)  # Moving along Y axis

        closure_rate = calculate_closure_rate(ship_a_pos, ship_a_vel, ship_b_pos, ship_b_vel)

        # No closure when moving perpendicular
        assert abs(closure_rate) < 1, f"Expected ~0 m/s, got {closure_rate}"

    def test_time_to_engagement_range(self, fleet_data):
        """Test calculation of time to reach engagement range."""
        # Ships 1000 km apart, closing at 20 km/s
        ship_a_pos = Vector3D(0, 0, 0)
        ship_a_vel = Vector3D(10_000, 0, 0)  # 10 km/s

        ship_b_pos = Vector3D(1_000_000, 0, 0)  # 1000 km
        ship_b_vel = Vector3D(-10_000, 0, 0)  # 10 km/s toward A

        # Engagement range of 500 km
        engagement_range_m = 500_000

        time_to_range = time_to_engagement_range(
            ship_a_pos, ship_a_vel, ship_b_pos, ship_b_vel, engagement_range_m
        )

        # Need to close 500 km at 20 km/s = 25 seconds
        expected_time = 25.0
        assert abs(time_to_range - expected_time) < 0.1, f"Expected {expected_time}s, got {time_to_range}s"

    def test_ships_separating_infinite_time(self, fleet_data):
        """Test that separating ships return infinite time to engagement."""
        # Both ships moving in same direction, ship B moving faster away
        ship_a_pos = Vector3D(0, 0, 0)
        ship_a_vel = Vector3D(5_000, 0, 0)

        ship_b_pos = Vector3D(1_000_000, 0, 0)
        ship_b_vel = Vector3D(10_000, 0, 0)  # Moving away from A

        time_to_range = time_to_engagement_range(
            ship_a_pos, ship_a_vel, ship_b_pos, ship_b_vel, 500_000
        )

        assert time_to_range == float('inf'), "Expected infinite time for separating ships"

    def test_destroyers_intercept(self, fleet_data):
        """Test intercept scenario with two destroyers from fleet data."""
        # Create two destroyers approaching each other
        destroyer_a = create_ship_from_fleet(
            fleet_data, "destroyer",
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(5_000, 0, 0),  # 5 km/s
            forward=Vector3D(1, 0, 0),
        )

        destroyer_b = create_ship_from_fleet(
            fleet_data, "destroyer",
            position=Vector3D(2_000_000, 0, 0),  # 2000 km away
            velocity=Vector3D(-5_000, 0, 0),  # 5 km/s toward A
            forward=Vector3D(-1, 0, 0),
        )

        # Calculate closure rate
        closure_rate = calculate_closure_rate(
            destroyer_a.position, destroyer_a.velocity,
            destroyer_b.position, destroyer_b.velocity
        )

        assert abs(closure_rate - 10_000) < 1, f"Expected 10 km/s closure, got {closure_rate}"

        # Spinal coiler range is 900 km
        time_to_range = time_to_engagement_range(
            destroyer_a.position, destroyer_a.velocity,
            destroyer_b.position, destroyer_b.velocity,
            engagement_range_m=900_000,
        )

        # Need to close 1100 km at 10 km/s = 110 seconds
        expected_time = 110.0
        assert abs(time_to_range - expected_time) < 1, f"Expected {expected_time}s, got {time_to_range}s"


# =============================================================================
# Scenario 2: Head-On Duel
# =============================================================================

class TestHeadOnDuel:
    """
    Test scenario: Destroyer vs Destroyer head-on duel.

    Tests nose armor protection and multiple hit ablation with spinal coilguns.
    """

    def test_nose_armor_protection(self, fleet_data, seeded_resolver):
        """Test that nose armor provides superior protection in head-on engagements."""
        destroyer_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        spinal_coiler = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")

        nose_armor = destroyer_armor.get_section(HitLocation.NOSE)
        lateral_armor = destroyer_armor.get_section(HitLocation.LATERAL)

        # Nose should have much higher protection than lateral
        assert nose_armor.protection > lateral_armor.protection
        # With Adamantane (baryonic_half_cm=115.8), protection_percent is lower but still significant
        # Destroyer nose at 151.2 cm Adamantane provides ~60% protection
        assert nose_armor.protection_percent > 50, f"Nose protection: {nose_armor.protection_percent}%"

        # Nose should have more hits to deplete
        # From fleet data: destroyer nose = 151.2 cm, lateral = 26.0 cm (Adamantane)
        assert nose_armor.thickness_cm > lateral_armor.thickness_cm * 5

    def test_multiple_hit_ablation_nose(self, fleet_data, seeded_rng):
        """Test armor degradation from multiple hits to nose."""
        # Create fresh armor for each test
        destroyer_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        spinal_coiler = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        resolver = CombatResolver(rng=seeded_rng)

        nose_armor = destroyer_armor.get_section(HitLocation.NOSE)
        initial_thickness = nose_armor.thickness_cm

        # Fire multiple shots at nose
        num_hits = 10
        for _ in range(num_hits):
            resolver.resolve_hit(
                spinal_coiler,
                destroyer_armor,
                location=HitLocation.NOSE,
            )

        # Armor should be degraded but not penetrated after 10 hits
        # (fleet data says 23 hits to deplete destroyer nose)
        assert nose_armor.thickness_cm < initial_thickness
        assert not nose_armor.is_penetrated(), "Nose should not be penetrated after 10 hits"

    def test_multiple_hit_ablation_lateral(self, fleet_data, seeded_rng):
        """Test that lateral armor degrades from sustained fire."""
        destroyer_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        spinal_coiler = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        resolver = CombatResolver(rng=seeded_rng)

        lateral_armor = destroyer_armor.get_section(HitLocation.LATERAL)
        initial_thickness = lateral_armor.thickness_cm

        # Fire sustained hits at lateral armor
        # Destroyer lateral: 26.0 cm Adamantane with high heat_of_vaporization (59.534 MJ/kg)
        # Adamantane ablates much slower than Titanium - test degradation rather than penetration
        num_hits = 50

        for _ in range(num_hits):
            resolver.resolve_hit(
                spinal_coiler,
                destroyer_armor,
                location=HitLocation.LATERAL,
            )

        # Armor should show significant degradation after sustained fire
        # With Adamantane's high heat_of_vaporization, ablation is slower
        final_thickness = lateral_armor.thickness_cm
        ablation_amount = initial_thickness - final_thickness

        # Should ablate at least 30% of armor after 50 hits
        ablation_percent = ablation_amount / initial_thickness * 100
        assert ablation_percent > 30, f"Only {ablation_percent:.1f}% ablated after {num_hits} hits"
        assert lateral_armor.thickness_cm > 0, "Some armor should remain after 50 hits"

    def test_head_on_engagement_simulation(self, fleet_data, seeded_rng):
        """Simulate a head-on engagement between two destroyers."""
        # Both destroyers facing each other
        destroyer_a_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        destroyer_b_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        spinal_coiler = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        resolver = CombatResolver(rng=seeded_rng)

        # Simulate 20 exchanges at 500 km range (within 900 km spinal range)
        distance_km = 500
        num_exchanges = 20

        a_hits = 0
        b_hits = 0
        a_penetrations = 0
        b_penetrations = 0

        for i in range(num_exchanges):
            # A fires at B (hits nose in head-on)
            result_a = resolver.resolve_attack(
                spinal_coiler,
                destroyer_b_armor,
                distance_km=distance_km,
                target_accel_g=2.0,  # Destroyer combat accel
            )
            if result_a.hit:
                a_hits += 1
                # In head-on, force nose hits
                resolver.resolve_hit(spinal_coiler, destroyer_b_armor, HitLocation.NOSE)
                if result_a.penetrated:
                    a_penetrations += 1

            # B fires at A (hits nose in head-on)
            result_b = resolver.resolve_attack(
                spinal_coiler,
                destroyer_a_armor,
                distance_km=distance_km,
                target_accel_g=2.0,
            )
            if result_b.hit:
                b_hits += 1
                resolver.resolve_hit(spinal_coiler, destroyer_a_armor, HitLocation.NOSE)
                if result_b.penetrated:
                    b_penetrations += 1

        # Both ships should have taken damage
        nose_a = destroyer_a_armor.get_section(HitLocation.NOSE)
        nose_b = destroyer_b_armor.get_section(HitLocation.NOSE)

        # With chip_resist=0.75 and flat_chipping=0.35, ablation per hit is:
        # 2.5 * 0.35 * (1 - 0.75) = 0.21875 cm per hit
        # Should still have significant armor after 20 hits
        assert nose_a.thickness_cm > 0, "Destroyer A nose should have armor remaining"
        assert nose_b.thickness_cm > 0, "Destroyer B nose should have armor remaining"


# =============================================================================
# Scenario 3: Flanking Attack
# =============================================================================

class TestFlankingAttack:
    """
    Test scenario: Corvette attacks cruiser from the side.

    Tests lateral armor vulnerability and rotation time impact on defense.
    """

    def test_lateral_armor_vulnerability(self, fleet_data, seeded_resolver):
        """Test that lateral armor is more vulnerable than nose armor."""
        cruiser_armor = create_ship_armor_from_fleet_data(fleet_data, "cruiser")
        spinal_coiler = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")

        nose_armor = cruiser_armor.get_section(HitLocation.NOSE)
        lateral_armor = cruiser_armor.get_section(HitLocation.LATERAL)

        # With Adamantane (baryonic_half_cm=115.8), protection percentages differ from Titanium
        # Cruiser nose: 240.7 cm Adamantane → ~76% protection
        # Cruiser lateral: 41.3 cm Adamantane → ~30% protection
        # Focus on relative difference rather than absolute percentages
        assert nose_armor.protection_percent > 70, f"Nose should have significant protection: {nose_armor.protection_percent}%"
        assert nose_armor.protection_percent > lateral_armor.protection_percent * 2, \
            f"Nose ({nose_armor.protection_percent:.1f}%) should be >2x lateral ({lateral_armor.protection_percent:.1f}%)"

        # Protection ratio shows vulnerability
        protection_ratio = lateral_armor.protection / nose_armor.protection
        assert protection_ratio < 0.5, f"Lateral protection ratio: {protection_ratio}"

    def test_cruiser_rotation_time(self, fleet_data):
        """Test that cruiser has significant rotation time to face attacker."""
        cruiser_data = fleet_data["ships"]["cruiser"]
        attitude = cruiser_data["attitude_control"]

        # Using thrust vectoring (combat maneuver)
        tv = attitude["thrust_vectoring"]
        time_90_tv = tv["time_to_rotate_90_deg_s"]

        # Using RCS (no thrust)
        rcs = attitude["rcs"]
        time_90_rcs = rcs["time_to_rotate_90_deg_s"]

        # Cruiser should take significant time to rotate
        # From fleet data: ~28s with thrust vectoring, ~206s with RCS
        assert time_90_tv > 25, f"90 deg rotation with TV: {time_90_tv}s"
        assert time_90_rcs > 200, f"90 deg rotation with RCS: {time_90_rcs}s"

    def test_flanking_engagement_simulation(self, fleet_data, seeded_rng):
        """Simulate corvette flanking attack on cruiser."""
        corvette_armor = create_ship_armor_from_fleet_data(fleet_data, "corvette")
        cruiser_armor = create_ship_armor_from_fleet_data(fleet_data, "cruiser")
        torpedo = create_weapon_from_fleet_data(fleet_data, "torpedo_launcher")
        resolver = CombatResolver(rng=seeded_rng)

        cruiser_data = fleet_data["ships"]["cruiser"]
        time_to_rotate_90 = cruiser_data["attitude_control"]["thrust_vectoring"]["time_to_rotate_90_deg_s"]

        # Corvette torpedo range is 2000 km, flies at ~10 km/s (estimated)
        # Time for torpedo to reach target from 1000 km = ~100s
        torpedo_flight_time = 100  # seconds (approximate)

        # Cruiser needs ~28s to rotate 90 degrees
        # If corvette attacks from side, cruiser can rotate to face torpedo
        # but first hits will be on lateral armor

        # Simulate 3 torpedo hits on lateral before rotation complete
        # Kinetic penetrator: 100 kg at typical 15 km/s impact = 0.5 * 100 * 15000^2 = 11.25 GJ
        # Create a weapon object with the calculated kinetic energy
        from dataclasses import replace
        torpedo_with_impact = replace(torpedo, kinetic_energy_gj=11.25)

        hits_before_rotation = 3
        total_damage_absorbed = 0.0
        for _ in range(hits_before_rotation):
            result = resolver.resolve_hit(
                torpedo_with_impact,
                cruiser_armor,
                location=HitLocation.LATERAL,
            )
            # Torpedoes do damage based on impact velocity
            assert result.hit
            total_damage_absorbed += result.damage_absorbed

        # 3 hits at ~11.25 GJ each should deliver significant damage
        # Lateral armor (Adamantane 41.3 cm) absorbs damage based on protection
        # Kinetic energy partially absorbed by armor, rest may penetrate or be deflected
        assert total_damage_absorbed > 5, f"Expected >5 GJ absorbed, got {total_damage_absorbed}"

        # Note: Kinetic penetrators ablate armor via kinetic energy transfer

    def test_rotation_advantage_calculation(self, fleet_data):
        """Calculate tactical advantage from rotation times."""
        corvette_data = fleet_data["ships"]["corvette"]
        cruiser_data = fleet_data["ships"]["cruiser"]

        corvette_rotate_90 = corvette_data["attitude_control"]["thrust_vectoring"]["time_to_rotate_90_deg_s"]
        cruiser_rotate_90 = cruiser_data["attitude_control"]["thrust_vectoring"]["time_to_rotate_90_deg_s"]

        # Corvette should rotate much faster than cruiser
        rotation_advantage = cruiser_rotate_90 / corvette_rotate_90

        # Cruiser takes ~28s, corvette takes ~12s -> ratio ~2.3
        assert rotation_advantage > 2.0, f"Rotation advantage: {rotation_advantage}x"


# =============================================================================
# Scenario 4: Torpedo Run
# =============================================================================

class TestTorpedoRun:
    """
    Test scenario: Corvette launches torpedo at battleship.

    Tests point defense engagement and evasion maneuvers.
    """

    def test_pd_engagement_range(self, fleet_data):
        """Test that PD has appropriate engagement range."""
        pd_laser = create_weapon_from_fleet_data(fleet_data, "pd_laser")

        # PD range should be 250 km (Terra Invicta value)
        assert pd_laser.range_km == 250, f"PD range: {pd_laser.range_km} km"
        assert pd_laser.cooldown_s == 5.0, "PD cooldown is 5 seconds (Terra Invicta value)"

    def test_torpedo_stats(self, fleet_data):
        """Verify torpedo statistics for tactical calculations."""
        torpedo = create_weapon_from_fleet_data(fleet_data, "torpedo_launcher")

        assert torpedo.is_missile, "Torpedo should be classified as missile (guided munition)"
        assert torpedo.range_km == 2500, f"Torpedo range: {torpedo.range_km} km (Trident specs)"
        # Kinetic penetrator - damage calculated at impact from mass*velocity^2, not stored here
        assert torpedo.kinetic_energy_gj == 0, f"Kinetic penetrator has no stored energy: {torpedo.kinetic_energy_gj} GJ"

    def test_evasion_command_generation(self, fleet_data, seeded_rng):
        """Test that tactical controller generates evasion commands for torpedoes."""
        battleship_physics = create_ship_from_fleet(
            fleet_data, "battleship",
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0),
        )

        # Create command module ship state
        weapons_status = {
            "spinal": {"status": "ready", "ammo": 450},
            "pd_forward": {"status": "ready", "type": "point_defense"},
        }

        battleship_cmd = create_command_ship_state(
            "battleship_01",
            battleship_physics,
            weapons_status,
        )

        # Create incoming torpedo threat
        incoming_torpedo = ThreatInfo(
            threat_id="torpedo_001",
            threat_type="torpedo",
            position=np.array([50_000.0, 0.0, 0.0]),  # 50 km away
            velocity=np.array([-5_000.0, 0.0, 0.0]),  # 5 km/s toward battleship
            estimated_time_to_impact=10.0,  # 10 seconds
            source_ship_id="corvette_01",
            can_be_intercepted=True,
        )

        # Create battle state
        battle_state = BattleState(
            timestamp=100.0,
            own_ship=battleship_cmd,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[incoming_torpedo],
            active_projectiles=[],
            recent_events=[],
            engagement_range=50_000,
            closing_rate=5_000,
            battle_duration=100.0,
        )

        # Create strategic controller
        strategic_controller = RuleBasedStrategicController(
            ship_id="battleship_01",
            decision_interval=10.0,
        )

        # Get commands for torpedo evasion
        commands = strategic_controller.decide(battle_state, [])

        # Should include an evade command
        evade_commands = [c for c in commands if isinstance(c, Evade)]
        assert len(evade_commands) > 0, "Should generate evade command for incoming torpedo"

    def test_pd_auto_engagement(self, fleet_data, seeded_rng):
        """Test tactical controller auto-engages torpedoes with PD."""
        battleship_physics = create_ship_from_fleet(
            fleet_data, "battleship",
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0),
        )

        weapons_status = {
            "pd_forward": {"status": "ready", "type": "point_defense"},
            "pd_aft": {"status": "ready", "type": "point_defense"},
        }

        battleship_cmd = create_command_ship_state(
            "battleship_01",
            battleship_physics,
            weapons_status,
        )

        incoming_torpedo = ThreatInfo(
            threat_id="torpedo_001",
            threat_type="torpedo",
            position=np.array([80_000.0, 0.0, 0.0]),  # Within 100km PD range
            velocity=np.array([-5_000.0, 0.0, 0.0]),
            estimated_time_to_impact=16.0,
            source_ship_id="corvette_01",
            can_be_intercepted=True,
        )

        battle_state = BattleState(
            timestamp=100.0,
            own_ship=battleship_cmd,
            enemy_ships=[],
            friendly_ships=[],
            incoming_threats=[incoming_torpedo],
            active_projectiles=[],
            recent_events=[],
            engagement_range=80_000,
            closing_rate=5_000,
            battle_duration=100.0,
        )

        # Create tactical controller with auto PD
        tactical_controller = TacticalController(
            ship_id="battleship_01",
            point_defense_auto=True,
        )

        # Run tactical update
        control_outputs, events = tactical_controller.update(battle_state, delta_time=1.0)

        # PD should auto-target the torpedo
        pd_targets = control_outputs.get('point_defense_targets', [])
        assert 'torpedo_001' in pd_targets, "PD should auto-target incoming torpedo"


# =============================================================================
# Scenario 5: Flip-and-Burn
# =============================================================================

class TestFlipAndBurn:
    """
    Test scenario: Ship decelerating to match velocity.

    Tests 180-degree rotation time and delta-v consumption.
    """

    def test_180_rotation_time(self, fleet_data):
        """Test rotation time for 180-degree flip maneuver."""
        # Test for multiple ship types
        ship_types = ["corvette", "destroyer", "cruiser", "battleship"]

        for ship_type in ship_types:
            ship_data = fleet_data["ships"][ship_type]
            attitude = ship_data["attitude_control"]

            # Thrust vectoring time (faster)
            tv_time = attitude["thrust_vectoring"]["time_to_rotate_180_deg_s"]

            # RCS time (slower)
            rcs_time = attitude["rcs"]["time_to_rotate_180_deg_s"]

            # All ships should be able to flip
            assert tv_time < 100, f"{ship_type} TV flip time: {tv_time}s"
            assert rcs_time > tv_time, f"{ship_type} RCS should be slower than TV"

    def test_delta_v_cost_for_rotation(self, fleet_data):
        """Test delta-v cost for 180-degree rotation during flip."""
        destroyer_data = fleet_data["ships"]["destroyer"]
        dv_cost = destroyer_data["attitude_control"]["thrust_vectoring"]["delta_v_cost"]

        # 180 degree rotation cost
        rotate_180_kps = dv_cost["rotate_180_deg_kps"]

        # Should be less than 1 km/s for cost-effective maneuver
        assert rotate_180_kps < 1.0, f"Rotation costs {rotate_180_kps} km/s"

    def test_flip_and_burn_simulation(self, fleet_data):
        """Simulate a complete flip-and-burn maneuver."""
        # Create destroyer moving at 50 km/s
        destroyer = create_ship_from_fleet(
            fleet_data, "destroyer",
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(50_000, 0, 0),  # 50 km/s
            forward=Vector3D(1, 0, 0),  # Facing direction of travel
        )

        initial_velocity = destroyer.velocity.magnitude
        initial_delta_v = destroyer.remaining_delta_v_ms()

        destroyer_data = fleet_data["ships"]["destroyer"]
        flip_time = destroyer_data["attitude_control"]["thrust_vectoring"]["time_to_rotate_180_deg_s"]

        # Phase 1: Flip (change orientation 180 degrees)
        # During flip, velocity doesn't change much (gimbal deflection is small)
        # Assume flip completes without significant position/velocity change

        # After flip, forward is now -X (opposite to velocity)
        flipped_destroyer = create_ship_from_fleet(
            fleet_data, "destroyer",
            position=destroyer.position,  # Same position
            velocity=destroyer.velocity,  # Same velocity
            forward=Vector3D(-1, 0, 0),  # Now facing backwards
        )

        # Phase 2: Burn to decelerate
        # At 2g (destroyer combat accel), how long to stop from 50 km/s?
        combat_accel = destroyer_data["performance"]["combat_acceleration_ms2"]
        time_to_stop = initial_velocity / combat_accel  # v = a*t -> t = v/a

        # Propagate with full thrust to decelerate
        trajectory = propagate_trajectory(
            flipped_destroyer,
            total_time=time_to_stop,
            dt=1.0,
            throttle=1.0,
        )

        final_state = trajectory[-1]

        # Velocity should be greatly reduced
        final_velocity = final_state.velocity.magnitude
        assert final_velocity < initial_velocity * 0.5, f"Final velocity: {final_velocity} m/s"

        # Delta-v should be consumed
        final_delta_v = final_state.remaining_delta_v_ms()
        delta_v_used = initial_delta_v - final_delta_v

        # Should have used approximately 50 km/s of delta-v
        assert delta_v_used > 40_000, f"Delta-v used: {delta_v_used} m/s"

    def test_flip_timing_verification(self, fleet_data):
        """Verify flip timing matches physics calculations."""
        destroyer_data = fleet_data["ships"]["destroyer"]
        attitude = destroyer_data["attitude_control"]

        # Get angular acceleration from fleet data
        angular_accel_deg_s2 = attitude["thrust_vectoring"]["angular_accel_deg_s2"]
        angular_accel_rad_s2 = math.radians(angular_accel_deg_s2)

        # Calculate expected time using physics module
        calculated_time = time_to_rotate(angular_accel_rad_s2, 180)

        # Compare with fleet data value
        fleet_time = attitude["thrust_vectoring"]["time_to_rotate_180_deg_s"]

        # Should be within 5% of each other
        tolerance = 0.05
        ratio = calculated_time / fleet_time
        assert abs(ratio - 1.0) < tolerance, f"Calculated: {calculated_time}s, Fleet: {fleet_time}s"


# =============================================================================
# Scenario 6: Capital Ship Battle
# =============================================================================

class TestCapitalShipBattle:
    """
    Test scenario: Dreadnought vs battleship + 2 destroyers.

    Tests multi-target engagement and armor degradation over time.
    """

    def test_multi_target_engagement_setup(self, fleet_data):
        """Verify fleet composition and weapon loadouts."""
        dreadnought_data = fleet_data["ships"]["dreadnought"]
        battleship_data = fleet_data["ships"]["battleship"]
        destroyer_data = fleet_data["ships"]["destroyer"]

        # Dreadnought weapons summary
        dread_weapons = dreadnought_data["weapons_summary"]
        assert dread_weapons["spinal_coilers"] == 1
        assert dread_weapons["heavy_coilguns"] == 5
        assert dread_weapons["point_defense"] == 4

        # Battleship weapons summary
        bs_weapons = battleship_data["weapons_summary"]
        assert bs_weapons["spinal_coilers"] == 1
        assert bs_weapons["heavy_coilguns"] == 3

        # Destroyer weapons summary
        dd_weapons = destroyer_data["weapons_summary"]
        assert dd_weapons["spinal_coilers"] == 1
        assert dd_weapons["coilguns"] == 1

    def test_dreadnought_vs_fleet_simulation(self, fleet_data, seeded_rng):
        """Simulate extended engagement between dreadnought and enemy fleet."""
        # Create armor instances
        dreadnought_armor = create_ship_armor_from_fleet_data(fleet_data, "dreadnought")
        battleship_armor = create_ship_armor_from_fleet_data(fleet_data, "battleship")
        destroyer_a_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        destroyer_b_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")

        # Create weapons
        spinal_coiler = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        heavy_coilgun = create_weapon_from_fleet_data(fleet_data, "heavy_coilgun_mk3")
        coilgun = create_weapon_from_fleet_data(fleet_data, "coilgun_mk3")

        resolver = CombatResolver(rng=seeded_rng)

        # Track battle statistics
        dread_damage_taken = 0.0
        battleship_damage_taken = 0.0
        destroyer_a_damage_taken = 0.0
        destroyer_b_damage_taken = 0.0

        # Simulate 60 seconds of combat (weapon cooldowns are ~10-45s)
        combat_time = 60  # seconds
        dt = 1.0  # 1 second time steps

        # Track weapon cooldowns
        dread_spinal_cd = 0
        dread_heavy_cds = [0, 0, 0, 0, 0]  # 5 heavy coilguns
        bs_spinal_cd = 0
        bs_heavy_cds = [0, 0, 0]  # 3 heavy coilguns
        dd_a_spinal_cd = 0
        dd_b_spinal_cd = 0

        distance_km = 400  # Combat range

        for t in range(int(combat_time)):
            # Dreadnought fires at battleship (primary target)
            if dread_spinal_cd <= 0:
                result = resolver.resolve_attack(
                    spinal_coiler, battleship_armor, distance_km, target_accel_g=1.0
                )
                if result.hit:
                    battleship_damage_taken += result.armor_ablation_cm
                dread_spinal_cd = spinal_coiler.cooldown_s

            # Heavy coilguns distributed across targets
            for i, cd in enumerate(dread_heavy_cds):
                if cd <= 0:
                    # Alternate targets
                    if i % 3 == 0:
                        target = battleship_armor
                        target_accel = 1.0
                    elif i % 3 == 1:
                        target = destroyer_a_armor
                        target_accel = 2.0
                    else:
                        target = destroyer_b_armor
                        target_accel = 2.0

                    result = resolver.resolve_attack(
                        heavy_coilgun, target, distance_km, target_accel_g=target_accel
                    )
                    if result.hit:
                        if target == battleship_armor:
                            battleship_damage_taken += result.armor_ablation_cm
                        elif target == destroyer_a_armor:
                            destroyer_a_damage_taken += result.armor_ablation_cm
                        else:
                            destroyer_b_damage_taken += result.armor_ablation_cm
                    dread_heavy_cds[i] = heavy_coilgun.cooldown_s

            # Enemy fleet fires at dreadnought
            if bs_spinal_cd <= 0:
                result = resolver.resolve_attack(
                    spinal_coiler, dreadnought_armor, distance_km, target_accel_g=0.75
                )
                if result.hit:
                    dread_damage_taken += result.armor_ablation_cm
                bs_spinal_cd = spinal_coiler.cooldown_s

            for i, cd in enumerate(bs_heavy_cds):
                if cd <= 0:
                    result = resolver.resolve_attack(
                        heavy_coilgun, dreadnought_armor, distance_km, target_accel_g=0.75
                    )
                    if result.hit:
                        dread_damage_taken += result.armor_ablation_cm
                    bs_heavy_cds[i] = heavy_coilgun.cooldown_s

            # Destroyer spinal coilers
            if dd_a_spinal_cd <= 0:
                result = resolver.resolve_attack(
                    spinal_coiler, dreadnought_armor, distance_km, target_accel_g=0.75
                )
                if result.hit:
                    dread_damage_taken += result.armor_ablation_cm
                dd_a_spinal_cd = spinal_coiler.cooldown_s

            if dd_b_spinal_cd <= 0:
                result = resolver.resolve_attack(
                    spinal_coiler, dreadnought_armor, distance_km, target_accel_g=0.75
                )
                if result.hit:
                    dread_damage_taken += result.armor_ablation_cm
                dd_b_spinal_cd = spinal_coiler.cooldown_s

            # Decrement cooldowns
            dread_spinal_cd = max(0, dread_spinal_cd - dt)
            dread_heavy_cds = [max(0, cd - dt) for cd in dread_heavy_cds]
            bs_spinal_cd = max(0, bs_spinal_cd - dt)
            bs_heavy_cds = [max(0, cd - dt) for cd in bs_heavy_cds]
            dd_a_spinal_cd = max(0, dd_a_spinal_cd - dt)
            dd_b_spinal_cd = max(0, dd_b_spinal_cd - dt)

        # Verify battle had significant action
        assert dread_damage_taken > 0, "Dreadnought should have taken damage"
        assert battleship_damage_taken > 0, "Battleship should have taken damage"

        # Dreadnought's superior armor should mean less proportional damage
        dread_initial_lateral = fleet_data["ships"]["dreadnought"]["armor"]["sections"]["lateral"]["thickness_cm"]
        bs_initial_lateral = fleet_data["ships"]["battleship"]["armor"]["sections"]["lateral"]["thickness_cm"]

        # The battle outcome depends on RNG, but both sides should sustain damage

    def test_armor_degradation_over_time(self, fleet_data, seeded_rng):
        """Test that armor degrades consistently over multiple engagements."""
        battleship_armor = create_ship_armor_from_fleet_data(fleet_data, "battleship")
        heavy_coilgun = create_weapon_from_fleet_data(fleet_data, "heavy_coilgun_mk3")
        resolver = CombatResolver(rng=seeded_rng)

        lateral_armor = battleship_armor.get_section(HitLocation.LATERAL)
        initial_thickness = lateral_armor.thickness_cm

        # Track degradation over multiple hits
        thicknesses = [initial_thickness]

        num_hits = 20
        for _ in range(num_hits):
            resolver.resolve_hit(
                heavy_coilgun,
                battleship_armor,
                location=HitLocation.LATERAL,
            )
            thicknesses.append(lateral_armor.thickness_cm)

        # Verify consistent degradation (armor should decrease monotonically)
        for i in range(1, len(thicknesses)):
            assert thicknesses[i] <= thicknesses[i-1], "Armor should not increase"

        # Verify some armor was ablated
        total_ablation = initial_thickness - thicknesses[-1]
        assert total_ablation > 0, "Should have ablated some armor"

    def test_firepower_concentration_effectiveness(self, fleet_data, seeded_rng):
        """Test the effectiveness of concentrating fire vs spreading it."""
        # Create two identical target armors
        target_concentrated = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        target_spread = create_ship_armor_from_fleet_data(fleet_data, "destroyer")

        heavy_coilgun = create_weapon_from_fleet_data(fleet_data, "heavy_coilgun_mk3")
        resolver = CombatResolver(rng=seeded_rng)

        # Concentrated fire: 10 hits all on lateral
        for _ in range(10):
            resolver.resolve_hit(
                heavy_coilgun,
                target_concentrated,
                location=HitLocation.LATERAL,
            )

        # Reset RNG for fair comparison
        resolver = CombatResolver(rng=random.Random(42))

        # Spread fire: hits distributed across all locations
        for i in range(10):
            location = [HitLocation.NOSE, HitLocation.LATERAL, HitLocation.TAIL][i % 3]
            resolver.resolve_hit(
                heavy_coilgun,
                target_spread,
                location=location,
            )

        # Concentrated fire should penetrate faster
        conc_lateral = target_concentrated.get_section(HitLocation.LATERAL)
        spread_lateral = target_spread.get_section(HitLocation.LATERAL)

        # Concentrated target's lateral should be more damaged
        assert conc_lateral.thickness_cm < spread_lateral.thickness_cm, \
            "Concentrated fire should damage lateral more"


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestSystemIntegration:
    """Additional tests to verify full system integration."""

    def test_physics_combat_command_integration(self, fleet_data, seeded_rng):
        """Test that all three modules work together in a combat scenario."""
        # Create physics states
        destroyer_physics = create_ship_from_fleet(
            fleet_data, "destroyer",
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(1, 0, 0),
        )

        target_physics = create_ship_from_fleet(
            fleet_data, "cruiser",
            position=Vector3D(300_000, 0, 0),  # 300 km away
            velocity=Vector3D(-1_000, 0, 0),  # Approaching
            forward=Vector3D(-1, 0, 0),
        )

        # Create combat entities
        destroyer_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        cruiser_armor = create_ship_armor_from_fleet_data(fleet_data, "cruiser")
        spinal_coiler = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        resolver = CombatResolver(rng=seeded_rng)

        # Create command module states
        destroyer_cmd = create_command_ship_state(
            "destroyer_01",
            destroyer_physics,
            {"spinal": {"status": "ready", "ammo": 450}},
        )

        target_cmd = create_command_ship_state(
            "cruiser_01",
            target_physics,
            {"spinal": {"status": "ready", "ammo": 450}},
        )

        # Create battle state
        battle_state = BattleState(
            timestamp=0.0,
            own_ship=destroyer_cmd,
            enemy_ships=[target_cmd],
            friendly_ships=[],
            incoming_threats=[],
            active_projectiles=[],
            recent_events=[],
            engagement_range=300_000,
            closing_rate=1_000,
            battle_duration=0.0,
        )

        # Strategic controller makes decision
        strategic = RuleBasedStrategicController("destroyer_01", decision_interval=10.0)
        commands = strategic.decide(battle_state, [])

        # Should have engagement commands
        engage_cmds = [c for c in commands if isinstance(c, Engage)]
        assert len(engage_cmds) > 0, "Should generate engage command"

        # Tactical controller executes
        tactical = TacticalController("destroyer_01")
        tactical.set_commands(commands)

        control_outputs, events = tactical.update(battle_state, delta_time=1.0)

        # Should try to face target
        assert control_outputs.get('target_heading') is not None, "Should set target heading"

        # Physics simulation advances
        new_physics = propagate_state(destroyer_physics, dt=1.0, throttle=0.5)

        # All systems worked together
        assert new_physics.position != destroyer_physics.position or \
               new_physics.velocity != destroyer_physics.velocity, \
               "Physics should have updated"

    def test_deterministic_battle_replay(self, fleet_data):
        """Test that the same seed produces identical battle results."""
        def run_battle(seed: int) -> tuple:
            rng = random.Random(seed)
            resolver = CombatResolver(rng=rng)

            destroyer_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
            spinal = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")

            results = []
            for _ in range(10):
                result = resolver.resolve_attack(spinal, destroyer_armor, 400, 2.0)
                results.append((result.hit, result.location, result.armor_ablation_cm))

            return tuple(results)

        # Run same battle twice
        results_a = run_battle(12345)
        results_b = run_battle(12345)

        # Should be identical
        assert results_a == results_b, "Same seed should produce identical results"

        # Different seed should produce different results
        results_c = run_battle(54321)
        assert results_a != results_c, "Different seed should produce different results"
