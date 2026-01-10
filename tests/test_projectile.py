#!/usr/bin/env python3
"""
Tests for Projectile Velocity Inheritance Mechanics

This test module validates that projectiles properly inherit velocity from
their launchers, which is critical for realistic space combat physics.

Key physics principles tested:
- Projectiles inherit shooter's velocity at launch
- Final velocity = shooter_velocity + muzzle_velocity (vector sum)
- Kinetic energy is calculated from REST FRAME velocity (total velocity)
- Torpedoes also inherit shooter velocity at launch

Examples:
- Ship at 50 km/s fires 10 km/s slug forward: slug = 60 km/s
- Ship at 50 km/s fires 10 km/s slug backward: slug = 40 km/s
- Ship at 50 km/s fires 10 km/s perpendicular: slug = sqrt(50^2+10^2) km/s
"""

import math
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics import Vector3D, ShipState
from projectile import (
    Projectile,
    KineticProjectile,
    ProjectileLauncher,
    calculate_kinetic_energy_gj,
    calculate_impact_velocity,
    KPS_TO_MS,
    GJ_TO_J,
)
from torpedo import Torpedo, TorpedoSpecs, TorpedoLauncher, GuidanceMode


# =============================================================================
# KINETIC PROJECTILE VELOCITY INHERITANCE TESTS
# =============================================================================

class TestKineticProjectileVelocityInheritance:
    """Test that kinetic projectiles properly inherit shooter velocity."""

    @pytest.mark.parametrize(
        "shooter_vel_kps,muzzle_vel_kps,fire_dir,expected_vel_kps",
        [
            # Forward fire: velocities add
            (50.0, 10.0, Vector3D(1, 0, 0), 60.0),
            # Backward fire: velocities subtract
            (50.0, 10.0, Vector3D(-1, 0, 0), 40.0),
            # Stationary shooter
            (0.0, 10.0, Vector3D(1, 0, 0), 10.0),
            # High velocity shooter
            (100.0, 10.0, Vector3D(1, 0, 0), 110.0),
            # Different muzzle velocities
            (50.0, 5.0, Vector3D(1, 0, 0), 55.0),
            (50.0, 20.0, Vector3D(1, 0, 0), 70.0),
        ],
        ids=[
            "forward_fire",
            "backward_fire",
            "stationary_shooter",
            "high_velocity_shooter",
            "slow_muzzle",
            "fast_muzzle",
        ]
    )
    def test_velocity_inheritance_aligned(
        self,
        shooter_vel_kps: float,
        muzzle_vel_kps: float,
        fire_dir: Vector3D,
        expected_vel_kps: float
    ):
        """Test velocity inheritance for aligned (forward/backward) fire."""
        shooter_velocity = Vector3D(shooter_vel_kps * KPS_TO_MS, 0, 0)

        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_direction=fire_dir,
            muzzle_velocity_kps=muzzle_vel_kps,
            mass_kg=25.0
        )

        assert abs(proj.speed_kps - expected_vel_kps) < 0.01, (
            f"Expected {expected_vel_kps} km/s, got {proj.speed_kps:.2f} km/s"
        )

    @pytest.mark.parametrize(
        "shooter_vel_kps,muzzle_vel_kps",
        [
            (50.0, 10.0),
            (30.0, 15.0),
            (100.0, 5.0),
            (10.0, 10.0),
        ],
        ids=["50_10", "30_15", "100_5", "10_10"]
    )
    def test_velocity_inheritance_perpendicular(
        self,
        shooter_vel_kps: float,
        muzzle_vel_kps: float
    ):
        """Test velocity inheritance for perpendicular fire."""
        shooter_velocity = Vector3D(shooter_vel_kps * KPS_TO_MS, 0, 0)
        fire_dir = Vector3D(0, 1, 0)  # Perpendicular to motion

        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_direction=fire_dir,
            muzzle_velocity_kps=muzzle_vel_kps,
            mass_kg=25.0
        )

        # Expected: sqrt(shooter_vel^2 + muzzle_vel^2)
        expected_vel_kps = math.sqrt(shooter_vel_kps**2 + muzzle_vel_kps**2)

        assert abs(proj.speed_kps - expected_vel_kps) < 0.01, (
            f"Expected {expected_vel_kps:.2f} km/s, got {proj.speed_kps:.2f} km/s"
        )

    def test_velocity_inheritance_diagonal_fire(self):
        """Test velocity inheritance for diagonal fire."""
        shooter_velocity = Vector3D(50000, 0, 0)  # 50 km/s in X
        fire_dir = Vector3D(1, 1, 0).normalized()  # 45 degrees
        muzzle_vel_kps = 10.0

        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_direction=fire_dir,
            muzzle_velocity_kps=muzzle_vel_kps,
            mass_kg=25.0
        )

        # Calculate expected velocity manually
        muzzle_component_x = fire_dir.x * muzzle_vel_kps * KPS_TO_MS
        muzzle_component_y = fire_dir.y * muzzle_vel_kps * KPS_TO_MS
        final_x = 50000 + muzzle_component_x
        final_y = muzzle_component_y
        expected_speed = math.sqrt(final_x**2 + final_y**2) / KPS_TO_MS

        assert abs(proj.speed_kps - expected_speed) < 0.01

    def test_launched_from_velocity_stored(self):
        """Test that shooter velocity is stored for reference."""
        shooter_velocity = Vector3D(50000, 10000, 0)

        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_direction=Vector3D(1, 0, 0),
            muzzle_velocity_kps=10.0,
            mass_kg=25.0
        )

        assert proj.launched_from_velocity.x == shooter_velocity.x
        assert proj.launched_from_velocity.y == shooter_velocity.y
        assert proj.launched_from_velocity.z == shooter_velocity.z

    def test_relative_velocity_equals_muzzle_velocity(self):
        """Test that velocity relative to shooter equals muzzle velocity."""
        shooter_velocity = Vector3D(50000, 10000, 5000)
        fire_dir = Vector3D(1, 0, 0)
        muzzle_vel_kps = 10.0

        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_direction=fire_dir,
            muzzle_velocity_kps=muzzle_vel_kps,
            mass_kg=25.0
        )

        relative_vel = proj.relative_velocity
        expected_relative = fire_dir * muzzle_vel_kps * KPS_TO_MS

        assert abs(relative_vel.x - expected_relative.x) < 0.01
        assert abs(relative_vel.y - expected_relative.y) < 0.01
        assert abs(relative_vel.z - expected_relative.z) < 0.01


# =============================================================================
# KINETIC ENERGY REST FRAME TESTS
# =============================================================================

class TestKineticEnergyRestFrame:
    """Test that kinetic energy is calculated from REST FRAME velocity."""

    @pytest.mark.parametrize(
        "shooter_vel_kps,muzzle_vel_kps,fire_dir,expected_speed_kps",
        [
            (50.0, 10.0, Vector3D(1, 0, 0), 60.0),
            (50.0, 10.0, Vector3D(-1, 0, 0), 40.0),
            (0.0, 10.0, Vector3D(1, 0, 0), 10.0),
        ],
        ids=["forward", "backward", "stationary"]
    )
    def test_kinetic_energy_from_rest_frame(
        self,
        shooter_vel_kps: float,
        muzzle_vel_kps: float,
        fire_dir: Vector3D,
        expected_speed_kps: float
    ):
        """Test that KE is calculated from total velocity, not muzzle velocity."""
        shooter_velocity = Vector3D(shooter_vel_kps * KPS_TO_MS, 0, 0)
        mass_kg = 25.0

        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_direction=fire_dir,
            muzzle_velocity_kps=muzzle_vel_kps,
            mass_kg=mass_kg
        )

        # Expected KE from REST FRAME velocity
        expected_speed_ms = expected_speed_kps * KPS_TO_MS
        expected_ke_j = 0.5 * mass_kg * expected_speed_ms**2
        expected_ke_gj = expected_ke_j / GJ_TO_J

        assert abs(proj.kinetic_energy_gj - expected_ke_gj) < 0.01, (
            f"Expected {expected_ke_gj:.2f} GJ, got {proj.kinetic_energy_gj:.2f} GJ"
        )

    def test_forward_vs_backward_energy_difference(self):
        """Test that forward fire has MORE kinetic energy than backward fire."""
        shooter_velocity = Vector3D(50000, 0, 0)  # 50 km/s
        muzzle_vel_kps = 10.0
        mass_kg = 25.0

        forward_proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_direction=Vector3D(1, 0, 0),
            muzzle_velocity_kps=muzzle_vel_kps,
            mass_kg=mass_kg
        )

        backward_proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_direction=Vector3D(-1, 0, 0),
            muzzle_velocity_kps=muzzle_vel_kps,
            mass_kg=mass_kg
        )

        # Forward: 60 km/s -> more energy
        # Backward: 40 km/s -> less energy
        assert forward_proj.kinetic_energy_gj > backward_proj.kinetic_energy_gj
        assert forward_proj.speed_kps == pytest.approx(60.0, abs=0.01)
        assert backward_proj.speed_kps == pytest.approx(40.0, abs=0.01)

        # Energy ratio should be (60/40)^2 = 2.25
        energy_ratio = forward_proj.kinetic_energy_gj / backward_proj.kinetic_energy_gj
        assert abs(energy_ratio - 2.25) < 0.01

    def test_kinetic_energy_utility_function(self):
        """Test the kinetic energy utility function."""
        mass_kg = 25.0
        velocity_kps = 60.0

        ke_gj = calculate_kinetic_energy_gj(mass_kg, velocity_kps)

        # Expected: 0.5 * 25 * (60000)^2 / 1e9 = 45 GJ
        expected = 0.5 * 25 * (60000**2) / 1e9
        assert abs(ke_gj - expected) < 0.01


# =============================================================================
# PROJECTILE LAUNCHER TESTS
# =============================================================================

class TestProjectileLauncher:
    """Test the ProjectileLauncher class."""

    def test_launch_kinetic_velocity_inheritance(self):
        """Test that launcher properly adds shooter velocity to projectile."""
        launcher = ProjectileLauncher(
            default_mass_kg=25.0,
            default_muzzle_velocity_kps=10.0
        )

        ship_state = ShipState(
            position=Vector3D.zero(),
            velocity=Vector3D(50000, 0, 0)  # 50 km/s
        )

        proj = launcher.launch_kinetic(
            shooter_state=ship_state,
            target_direction=Vector3D(1, 0, 0),
            current_time=0.0
        )

        assert proj is not None
        assert proj.speed_kps == pytest.approx(60.0, abs=0.01)
        assert proj.launched_from_velocity.x == 50000

    @pytest.mark.parametrize(
        "shooter_vel_kps,expected_proj_vel_kps",
        [
            (0.0, 10.0),
            (25.0, 35.0),
            (50.0, 60.0),
            (100.0, 110.0),
        ],
        ids=["stationary", "25kps", "50kps", "100kps"]
    )
    def test_launcher_various_shooter_velocities(
        self,
        shooter_vel_kps: float,
        expected_proj_vel_kps: float
    ):
        """Test launcher with different shooter velocities."""
        launcher = ProjectileLauncher(
            default_mass_kg=25.0,
            default_muzzle_velocity_kps=10.0
        )

        ship_state = ShipState(
            position=Vector3D.zero(),
            velocity=Vector3D(shooter_vel_kps * KPS_TO_MS, 0, 0)
        )

        proj = launcher.launch_kinetic(
            shooter_state=ship_state,
            target_direction=Vector3D(1, 0, 0),
            current_time=0.0
        )

        assert proj is not None
        assert proj.speed_kps == pytest.approx(expected_proj_vel_kps, abs=0.01)

    def test_launcher_cooldown(self):
        """Test that launcher respects cooldown."""
        launcher = ProjectileLauncher(cooldown_seconds=5.0)

        ship_state = ShipState(
            position=Vector3D.zero(),
            velocity=Vector3D(50000, 0, 0)
        )

        # First shot should work
        proj1 = launcher.launch_kinetic(
            shooter_state=ship_state,
            target_direction=Vector3D(1, 0, 0),
            current_time=0.0
        )
        assert proj1 is not None

        # Second shot during cooldown should fail
        proj2 = launcher.launch_kinetic(
            shooter_state=ship_state,
            target_direction=Vector3D(1, 0, 0),
            current_time=3.0
        )
        assert proj2 is None

        # Third shot after cooldown should work
        proj3 = launcher.launch_kinetic(
            shooter_state=ship_state,
            target_direction=Vector3D(1, 0, 0),
            current_time=10.0
        )
        assert proj3 is not None


# =============================================================================
# TORPEDO VELOCITY INHERITANCE TESTS
# =============================================================================

class TestTorpedoVelocityInheritance:
    """Test that torpedoes properly inherit shooter velocity."""

    def test_torpedo_launch_inherits_velocity(self):
        """Test that torpedo inherits shooter velocity at launch."""
        specs = TorpedoSpecs()
        launcher = TorpedoLauncher(specs=specs)

        shooter_velocity = Vector3D(50000, 0, 0)  # 50 km/s

        torpedo = launcher.launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_id="target_001",
            target_position=Vector3D(500000, 0, 0),
            target_velocity=Vector3D(-5000, 0, 0),
            current_time=0.0
        )

        assert torpedo is not None
        # Torpedo should have shooter's velocity
        assert torpedo.velocity.x == 50000
        assert torpedo.velocity.y == 0
        assert torpedo.velocity.z == 0

    def test_torpedo_launched_from_velocity_stored(self):
        """Test that torpedo stores the shooter velocity for reference."""
        specs = TorpedoSpecs()
        launcher = TorpedoLauncher(specs=specs)

        shooter_velocity = Vector3D(50000, 10000, 5000)

        torpedo = launcher.launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_id="target_001",
            target_position=Vector3D(500000, 0, 0),
            target_velocity=Vector3D(-5000, 0, 0),
            current_time=0.0
        )

        assert torpedo is not None
        assert torpedo.launched_from_velocity.x == 50000
        assert torpedo.launched_from_velocity.y == 10000
        assert torpedo.launched_from_velocity.z == 5000

    @pytest.mark.parametrize(
        "shooter_vel_x,shooter_vel_y,shooter_vel_z",
        [
            (0, 0, 0),
            (50000, 0, 0),
            (0, 50000, 0),
            (30000, 40000, 0),
            (10000, 20000, 30000),
        ],
        ids=["stationary", "x_only", "y_only", "xy_diagonal", "xyz_3d"]
    )
    def test_torpedo_velocity_inheritance_various_directions(
        self,
        shooter_vel_x: float,
        shooter_vel_y: float,
        shooter_vel_z: float
    ):
        """Test torpedo velocity inheritance with various shooter directions."""
        specs = TorpedoSpecs()
        launcher = TorpedoLauncher(specs=specs)

        shooter_velocity = Vector3D(shooter_vel_x, shooter_vel_y, shooter_vel_z)

        torpedo = launcher.launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=shooter_velocity,
            target_id="target_001",
            target_position=Vector3D(500000, 0, 0),
            target_velocity=Vector3D.zero(),
            current_time=0.0
        )

        assert torpedo is not None
        assert torpedo.velocity.x == shooter_vel_x
        assert torpedo.velocity.y == shooter_vel_y
        assert torpedo.velocity.z == shooter_vel_z
        assert torpedo.launched_from_velocity.x == shooter_vel_x
        assert torpedo.launched_from_velocity.y == shooter_vel_y
        assert torpedo.launched_from_velocity.z == shooter_vel_z

    def test_torpedo_thrust_adds_to_inherited_velocity(self):
        """Test that torpedo thrust adds to the inherited velocity."""
        specs = TorpedoSpecs()

        # Create torpedo with inherited velocity
        torpedo = Torpedo(
            specs=specs,
            position=Vector3D.zero(),
            velocity=Vector3D(50000, 0, 0),  # 50 km/s inherited
            target_id="target_001",
            launched_from_velocity=Vector3D(50000, 0, 0)
        )

        initial_speed = torpedo.velocity.magnitude

        # Apply thrust in the forward direction
        thrust_direction = Vector3D(1, 0, 0)
        torpedo.apply_thrust(thrust_direction, dt_seconds=1.0)

        # Speed should have increased
        final_speed = torpedo.velocity.magnitude
        assert final_speed > initial_speed


# =============================================================================
# IMPACT VELOCITY TESTS
# =============================================================================

class TestImpactVelocity:
    """Test impact velocity calculations."""

    def test_impact_velocity_approaching_target(self):
        """Test impact velocity when target is approaching."""
        shooter_velocity = Vector3D(50000, 0, 0)  # 50 km/s forward
        target_velocity = Vector3D(-50000, 0, 0)  # 50 km/s toward shooter
        fire_dir = Vector3D(1, 0, 0)
        muzzle_vel_kps = 10.0

        impact_vel = calculate_impact_velocity(
            shooter_velocity, muzzle_vel_kps, fire_dir, target_velocity
        )

        # Projectile: 50 + 10 = 60 km/s forward
        # Target: 50 km/s backward
        # Relative: 60 - (-50) = 110 km/s
        expected = 110000  # 110 km/s in m/s
        assert abs(impact_vel - expected) < 1.0

    def test_impact_velocity_receding_target(self):
        """Test impact velocity when target is receding."""
        shooter_velocity = Vector3D(50000, 0, 0)  # 50 km/s
        target_velocity = Vector3D(30000, 0, 0)  # 30 km/s same direction
        fire_dir = Vector3D(1, 0, 0)
        muzzle_vel_kps = 10.0

        impact_vel = calculate_impact_velocity(
            shooter_velocity, muzzle_vel_kps, fire_dir, target_velocity
        )

        # Projectile: 60 km/s, Target: 30 km/s same direction
        # Relative: 60 - 30 = 30 km/s
        expected = 30000
        assert abs(impact_vel - expected) < 1.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestProjectileSystemIntegration:
    """Integration tests for the projectile system."""

    def test_projectile_update_position(self):
        """Test that projectile position updates correctly."""
        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=Vector3D(50000, 0, 0),
            target_direction=Vector3D(1, 0, 0),
            muzzle_velocity_kps=10.0,
            mass_kg=25.0
        )

        # After 1 second at 60 km/s, should be 60 km away
        proj.update(1.0)

        expected_pos = 60000  # 60 km in meters
        assert abs(proj.position.x - expected_pos) < 1.0

    def test_projectile_distance_calculation(self):
        """Test distance to target calculation."""
        proj = KineticProjectile.from_launch(
            shooter_position=Vector3D.zero(),
            shooter_velocity=Vector3D(50000, 0, 0),
            target_direction=Vector3D(1, 0, 0),
            muzzle_velocity_kps=10.0,
            mass_kg=25.0
        )

        target = Vector3D(100000, 0, 0)  # 100 km away
        distance = proj.distance_to(target)

        assert abs(distance - 100000) < 1.0

    def test_full_engagement_scenario(self):
        """Test a complete engagement scenario with velocity inheritance."""
        # Ship moving at 50 km/s fires at approaching target
        launcher = ProjectileLauncher(
            default_mass_kg=25.0,
            default_muzzle_velocity_kps=10.0
        )

        ship = ShipState(
            position=Vector3D.zero(),
            velocity=Vector3D(50000, 0, 0)
        )

        target_pos = Vector3D(100000, 0, 0)
        target_vel = Vector3D(-20000, 0, 0)  # Approaching at 20 km/s

        # Calculate intercept direction
        intercept_dir = launcher.calculate_intercept_direction(
            ship, target_pos, target_vel
        )

        assert intercept_dir is not None

        # Launch projectile
        proj = launcher.launch_kinetic(
            shooter_state=ship,
            target_direction=intercept_dir,
            current_time=0.0
        )

        assert proj is not None
        # Projectile should have inherited ship velocity + muzzle velocity
        assert proj.speed_kps > 50.0  # Must be faster than ship


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
