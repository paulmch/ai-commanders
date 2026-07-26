"""
Regression tests for audited physics-partition defect fixes.

Covers fixes in: src/physics.py, src/maneuvers.py, src/geometry.py,
src/targeting.py, src/projectile.py.
"""

import math

import pytest

from src.physics import ShipState, Vector3D, apply_thrust, propagate_state
from src.maneuvers import (
    AccelerationBurn,
    BreakTurn,
    ManeuverPlanner,
    ManeuverStatus,
    RotateToFace,
)
from src.geometry import ShipGeometry, WeaponType
from src.targeting import LeadCalculator
from src.projectile import ProjectileLauncher


def make_test_ship(**overrides) -> ShipState:
    """Small, agile test ship with well-conditioned numbers."""
    defaults = dict(
        mass_kg=10_000.0,
        dry_mass_kg=5_000.0,
        propellant_kg=5_000.0,
        thrust_n=1_000_000.0,
        exhaust_velocity_ms=10_256_000.0,
        moment_of_inertia_kg_m2=1_000_000.0,
    )
    defaults.update(overrides)
    return ShipState(**defaults)


# =============================================================================
# Finding 1: TVC rotation direction (nose must swing AWAY from thrust side)
# =============================================================================

class TestThrustVectoringDirection:
    def test_pitch_gimbal_nose_moves_opposite_lateral_thrust(self):
        """Positive pitch gimbal: lateral force one way, nose the other way.

        A rear-mounted engine whose thrust deflects toward +up pushes the
        tail toward +up, so the nose must pitch toward -up.
        """
        state = make_test_ship()
        for _ in range(30):
            state = propagate_state(state, 0.01, throttle=1.0, gimbal_pitch_deg=3.0)

        assert state.velocity.z != pytest.approx(0.0), "gimbal must produce lateral thrust"
        assert state.forward.z != pytest.approx(0.0), "gimbal must rotate the nose"
        # Opposite signs: nose swings away from the deflected-thrust side
        assert state.velocity.z * state.forward.z < 0, (
            f"nose (fwd.z={state.forward.z:.4f}) must rotate opposite the "
            f"lateral thrust (vel.z={state.velocity.z:.4f})"
        )

    def test_yaw_gimbal_nose_moves_opposite_lateral_thrust(self):
        state = make_test_ship()
        for _ in range(30):
            state = propagate_state(state, 0.01, throttle=1.0, gimbal_yaw_deg=3.0)

        assert state.velocity.y != pytest.approx(0.0)
        assert state.forward.y != pytest.approx(0.0)
        assert state.velocity.y * state.forward.y < 0, (
            "nose must rotate opposite the lateral thrust for yaw as well"
        )


# =============================================================================
# Finding 3: orientation stays orthonormal under combined rotation
# =============================================================================

class TestOrientationOrthogonality:
    def test_forward_up_stay_orthonormal_under_combined_rotation(self):
        state = ShipState()
        state.angular_velocity = Vector3D(0.05, 0.1, 0.08)

        for _ in range(1000):
            state = propagate_state(state, 1.0)

        assert abs(state.forward.dot(state.up)) < 1e-6, (
            f"forward/up skewed: dot = {state.forward.dot(state.up)}"
        )
        assert state.forward.magnitude == pytest.approx(1.0, abs=1e-9)
        assert state.up.magnitude == pytest.approx(1.0, abs=1e-9)


# =============================================================================
# Finding 4: no free delta-v when propellant runs out mid-step
# =============================================================================

class TestPropellantClampedThrust:
    def test_final_step_delta_v_bounded_by_rocket_equation(self):
        """With 0.5 kg left and a 10 s step, dv must obey Tsiolkovsky."""
        state = make_test_ship(mass_kg=5_000.5, propellant_kg=0.5)
        accel, consumed = apply_thrust(state, dt=10.0, throttle=1.0)
        dv = accel.magnitude * 10.0

        allowed = state.exhaust_velocity_ms * math.log(5_000.5 / 5_000.0)
        assert consumed == pytest.approx(0.5)
        # Small tolerance for the average-mass discretization; the old code
        # overshot this bound by a factor of ~50.
        assert dv <= allowed * 1.001, f"dv {dv} exceeds Tsiolkovsky bound {allowed}"
        assert dv == pytest.approx(allowed, rel=1e-3)

    def test_depletion_burn_is_timestep_independent(self):
        """Burning to depletion must give the same final dv at dt=1 and dt=10."""
        def burn_out(dt: float) -> float:
            state = make_test_ship(mass_kg=5_010.0, propellant_kg=10.0)
            for _ in range(1000):
                if state.propellant_kg <= 0:
                    break
                accel, consumed = apply_thrust(state, dt=dt, throttle=1.0)
                state.velocity = state.velocity + accel * dt
                state.propellant_kg -= consumed
                state.mass_kg -= consumed
            return state.velocity.magnitude

        dv_fine = burn_out(1.0)
        dv_coarse = burn_out(10.0)
        allowed = 10_256_000.0 * math.log(5_010.0 / 5_000.0)

        assert dv_coarse <= allowed * 1.001
        assert dv_coarse == pytest.approx(dv_fine, rel=1e-3)


# =============================================================================
# Finding 9: coordinate system consistency (right-handed, right = starboard)
# =============================================================================

class TestCoordinateHandedness:
    def test_default_right_is_minus_y_and_orthonormal(self):
        """Documented convention: X fwd, Y port, Z up; starboard = -Y."""
        state = ShipState()
        right = state.right
        assert right.x == pytest.approx(0.0, abs=1e-12)
        assert right.y == pytest.approx(-1.0, abs=1e-12)
        assert right.z == pytest.approx(0.0, abs=1e-12)
        # Right-handed triple: forward x up = right, up x forward = -right
        assert Vector3D.unit_x().cross(Vector3D.unit_y()).z == pytest.approx(1.0)


# =============================================================================
# Finding 2: BreakTurn burn phase actually fires after a realistic slow turn
# =============================================================================

class TestBreakTurnBurnPhase:
    @staticmethod
    def _run_break_turn(turn_rate_deg_s: float, burn_duration: float, dt: float = 1.0):
        """Drive a BreakTurn with the sim rotating at a realistic rate.

        Returns (throttle_on_seconds, completed).
        """
        ship = make_test_ship()
        maneuver = BreakTurn(
            turn_angle_deg=90.0, turn_direction="right",
            burn_duration=burn_duration, throttle=1.0,
        )
        target_dir = maneuver._calculate_target_direction(ship)

        throttle_on = 0.0
        for _ in range(500):
            result = maneuver.execute_step(ship, dt)
            if result.status != ManeuverStatus.IN_PROGRESS:
                break
            if result.throttle > 0:
                throttle_on += dt
            else:
                # Rotate toward the commanded direction at the given rate
                angle = ship.forward.angle_to(target_dir)
                step = min(math.radians(turn_rate_deg_s) * dt, angle)
                if angle > 1e-9 and step > 0:
                    axis = ship.forward.cross(target_dir).normalized()
                    ship.forward = ship.forward.rotate_around_axis(axis, step).normalized()
        return throttle_on, maneuver.status == ManeuverStatus.COMPLETED

    def test_burn_fires_for_full_duration_with_slow_turner(self):
        """At a destroyer-like 4.4 deg/s the old code skipped the burn entirely."""
        throttle_on, completed = self._run_break_turn(
            turn_rate_deg_s=4.4, burn_duration=10.0, dt=1.0
        )
        assert completed
        assert throttle_on == pytest.approx(10.0, abs=1.0), (
            f"burn phase ran {throttle_on}s of throttle instead of ~10s"
        )

    def test_burn_duration_independent_of_turn_rate(self):
        fast, _ = self._run_break_turn(turn_rate_deg_s=15.0, burn_duration=10.0)
        slow, _ = self._run_break_turn(turn_rate_deg_s=2.0, burn_duration=10.0)
        assert fast == pytest.approx(slow, abs=1.0)


# =============================================================================
# Finding 5: intercept time for receding targets uses the correct root
# =============================================================================

class TestInterceptTimeRecedingTarget:
    def test_separation_is_closed_at_returned_time(self):
        pursuer = make_test_ship()  # at rest at origin
        accel = pursuer.max_acceleration_ms2()
        distance = 100_000.0  # 100 km ahead
        recede_speed = 100.0  # m/s, directly away

        t = ManeuverPlanner.calculate_intercept_time(
            pursuer,
            target_position=Vector3D(distance, 0, 0),
            target_velocity=Vector3D(recede_speed, 0, 0),
        )

        assert math.isfinite(t) and t > 0
        # Kinematics: distance closed by pursuer minus target's escape must
        # equal the initial separation at the intercept time.
        remaining = distance + recede_speed * t - 0.5 * accel * t * t
        assert remaining == pytest.approx(0.0, abs=1.0), (
            f"at returned t={t:.2f}s the ships are still {remaining:.0f} m apart"
        )


# =============================================================================
# Findings 6 & 7: lead/intercept solutions account for shooter velocity
# (projectiles inherit the shooter's velocity)
# =============================================================================

def _miss_distance_km(shooter_pos, shooter_vel, target_pos, target_vel,
                      muzzle_dir, muzzle_speed):
    """Closest approach of a velocity-inheriting projectile to the target.

    All quantities in km / km/s. Samples the intercept window finely.
    """
    proj_vel = shooter_vel + muzzle_dir * muzzle_speed
    best = float("inf")
    for i in range(1, 40001):
        t = i * 0.001  # up to 40 s
        proj = shooter_pos + proj_vel * t
        tgt = target_pos + target_vel * t
        best = min(best, (proj - tgt).magnitude)
    return best


class TestLeadCalculatorShooterVelocity:
    def test_moving_shooter_lead_hits_with_inherited_velocity(self):
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 5, 0)  # 5 km/s perpendicular
        target_pos = Vector3D(100, 0, 0)  # stationary, 100 km away
        target_vel = Vector3D(0, 0, 0)
        speed = 10.0  # km/s

        lead = LeadCalculator.calculate_lead(
            shooter_pos, shooter_vel, target_pos, target_vel, speed
        )
        muzzle_dir = (lead - shooter_pos).normalized()

        miss = _miss_distance_km(
            shooter_pos, shooter_vel, target_pos, target_vel, muzzle_dir, speed
        )
        assert miss < 0.2, f"velocity-inheriting projectile misses by {miss:.2f} km"

    def test_lead_depends_on_shooter_velocity(self):
        """A moving shooter must get a different aim point than a static one."""
        target_pos = Vector3D(100, 0, 0)
        static = LeadCalculator.calculate_lead(
            Vector3D(0, 0, 0), Vector3D(0, 0, 0),
            target_pos, Vector3D(0, 0, 0), 10.0,
        )
        moving = LeadCalculator.calculate_lead(
            Vector3D(0, 0, 0), Vector3D(0, 5, 0),
            target_pos, Vector3D(0, 0, 0), 10.0,
        )
        assert (static - moving).magnitude > 1.0, (
            "shooter velocity had no effect on the lead solution"
        )

    def test_lead_with_acceleration_moving_shooter(self):
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 5, 0)
        target_pos = Vector3D(100, 0, 0)
        target_vel = Vector3D(0, 0, 0)

        lead = LeadCalculator.calculate_lead_with_acceleration(
            shooter_pos, shooter_vel, target_pos, target_vel,
            Vector3D(0, 0, 0), 10.0,
        )
        muzzle_dir = (lead - shooter_pos).normalized()
        miss = _miss_distance_km(
            shooter_pos, shooter_vel, target_pos, target_vel, muzzle_dir, 10.0
        )
        assert miss < 0.2, f"accelerating-lead aim misses by {miss:.2f} km"


class TestProjectileInterceptDirection:
    def test_moving_shooter_stationary_target_hits(self):
        launcher = ProjectileLauncher(default_muzzle_velocity_kps=10.0)
        shooter = make_test_ship()
        shooter.velocity = Vector3D(0, 5000.0, 0)  # 5 km/s perpendicular (m/s)
        target_position = Vector3D(100_000.0, 0, 0)  # 100 km (meters)
        target_velocity = Vector3D(0, 0, 0)

        direction = launcher.calculate_intercept_direction(
            shooter, target_position, target_velocity
        )
        assert direction is not None
        # Must compensate for own motion: aim has a -y component
        assert direction.y < 0, "aim must compensate for shooter's +y velocity"

        miss = _miss_distance_km(
            shooter.position * 0.001, shooter.velocity * 0.001,
            target_position * 0.001, target_velocity * 0.001,
            direction, 10.0,
        )
        assert miss < 0.2, f"inherited-velocity projectile misses by {miss:.2f} km"


# =============================================================================
# Finding 8: cone surface normals tilt the correct way
# =============================================================================

class TestSurfaceNormals:
    @pytest.fixture
    def geometry(self):
        return ShipGeometry(
            length_m=100.0, radius_m=12.5,
            nose_cone_length_m=15.0, engine_section_length_m=15.0,
        )

    def test_nose_cone_normal_tilts_forward(self, geometry):
        normal = geometry._calculate_surface_normal(
            axial_position_m=5.0, radial_angle_deg=0.0,
            ship_fwd=Vector3D.unit_x(), ship_up=Vector3D.unit_z(),
            ship_right=Vector3D(0, -1, 0),
        )
        assert normal.magnitude == pytest.approx(1.0, abs=1e-9)
        assert normal.x > 0, f"nose normal must tilt forward, got x={normal.x}"
        assert normal.z > 0, "nose normal must point radially outward"

    def test_engine_cone_normal_tilts_backward(self, geometry):
        normal = geometry._calculate_surface_normal(
            axial_position_m=95.0, radial_angle_deg=0.0,
            ship_fwd=Vector3D.unit_x(), ship_up=Vector3D.unit_z(),
            ship_right=Vector3D(0, -1, 0),
        )
        assert normal.x < 0, f"engine normal must tilt backward, got x={normal.x}"
        assert normal.z > 0

    def test_cylinder_normal_purely_radial(self, geometry):
        normal = geometry._calculate_surface_normal(
            axial_position_m=50.0, radial_angle_deg=0.0,
            ship_fwd=Vector3D.unit_x(), ship_up=Vector3D.unit_z(),
            ship_right=Vector3D(0, -1, 0),
        )
        assert normal.x == pytest.approx(0.0, abs=1e-9)
        assert normal.z == pytest.approx(1.0, abs=1e-9)


# =============================================================================
# Finding 12: firing-arc classification
# =============================================================================

class TestFiringArcClassification:
    @pytest.fixture
    def geometry(self):
        return ShipGeometry(
            length_m=100.0, radius_m=12.5,
            nose_cone_length_m=15.0, engine_section_length_m=15.0,
        )

    def test_heavy_siege_coiler_is_limited_arc_not_turret(self, geometry):
        arc = geometry.get_weapon_firing_arc(Vector3D(50, 0, 0), "heavy_siege_coiler_mk3")
        assert arc.weapon_type == WeaponType.SPINAL
        assert not arc.can_fire_full_sphere
        assert arc.half_angle_deg <= 20.0, (
            f"fixed spinal mount got a {arc.half_angle_deg} deg arc"
        )

    def test_spinal_coiler_stays_spinal(self, geometry):
        arc = geometry.get_weapon_firing_arc(Vector3D(50, 0, 0), "spinal_coiler_mk3")
        assert arc.weapon_type == WeaponType.SPINAL
        assert arc.half_angle_deg <= 5.0

    def test_pd_laser_full_sphere(self, geometry):
        arc = geometry.get_weapon_firing_arc(Vector3D(0, 0, 12.5), "pd_laser")
        assert arc.weapon_type == WeaponType.POINT_DEFENSE
        assert arc.can_fire_full_sphere

    def test_generic_laser_not_full_sphere(self, geometry):
        """'laser' alone must no longer grant full-sphere PD coverage."""
        arc = geometry.get_weapon_firing_arc(Vector3D(0, 0, 12.5), "siege_laser_battery")
        assert not arc.can_fire_full_sphere

    def test_turret_coilgun_unchanged(self, geometry):
        arc = geometry.get_weapon_firing_arc(Vector3D(0, 5, 0), "heavy_coilgun_mk3")
        assert arc.weapon_type == WeaponType.TURRET


# =============================================================================
# Finding 10: rotation delta-v estimates consistent with their time model
# =============================================================================

class TestRotationDeltaVEstimate:
    def test_estimate_reflects_engine_on_thrust_vectoring(self):
        ship = ShipState()  # default corvette
        maneuver = RotateToFace(target_position=Vector3D(0, 1_000_000.0, 0))

        time_est = maneuver.estimate_completion_time(ship)
        dv_est = maneuver.estimate_delta_v_cost(ship)

        assert time_est > 0
        # The time model assumes the main engine at 0.3 throttle for the
        # whole turn; the dv estimate must be the same order of magnitude.
        engine_on_dv = 0.3 * ship.max_acceleration_ms2() * time_est
        assert dv_est == pytest.approx(engine_on_dv, rel=0.01), (
            f"dv estimate {dv_est:.1f} m/s inconsistent with the model's "
            f"engine-on cost {engine_on_dv:.1f} m/s"
        )
        # Regression: the old flat 0.1 m/s-per-second estimate
        assert dv_est > 10 * (time_est * 0.1)


# =============================================================================
# Finding 11: suggest_evasion burns perpendicular to the threat axis
# =============================================================================

class TestSuggestEvasionPerpendicular:
    def test_calculated_evasion_is_perpendicular_burn(self):
        ship = make_test_ship()
        # Distant, slowly-closing threat -> "time for calculated evasion"
        threat_position = Vector3D(500_000.0, 0, 0)
        threat_velocity = Vector3D(-1_000.0, 0, 0)  # 500 s to impact

        maneuver = ManeuverPlanner.suggest_evasion(ship, threat_position, threat_velocity)

        assert isinstance(maneuver, AccelerationBurn), (
            f"expected a perpendicular AccelerationBurn, got {type(maneuver).__name__}"
        )
        threat_dir = threat_velocity.normalized()
        assert abs(maneuver.direction.dot(threat_dir)) < 1e-6, (
            "evasion burn must be perpendicular to the threat approach axis"
        )
        # Chosen sign must not close the range toward the threat
        rel_pos = threat_position - ship.position
        assert maneuver.direction.dot(rel_pos) <= 1e-6
